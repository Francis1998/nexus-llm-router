"""Tests for the Anthropic Messages API compatible ``/v1/messages`` endpoints."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as main
from adapters.base import BaseProviderAdapter, ProviderError
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from api.anthropic_messages import StopSequenceFilter, apply_stop_sequences
from cache.response_cache import ResponseCache
from router.analyzer import RequestAnalyzer
from router.config import RouterSettings
from router.engine import NexusRouter, RoutingFailedError
from router.model_ids import ANTHROPIC_SAFETY_MODEL, GEMINI_FLASH_MODEL
from router.schemas import ChatMessage, ProviderResponse, RouterRequest, RouterResponse
from safety.budget import BudgetExceededError
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.key_fingerprint import fingerprint_api_key
from safety.rate_limiter import RateLimitExceededError
from safety.spend_ledger import SpendLedger
from safety.tenant_rate_limiter import TenantRateLimiter

PROVIDERS = ("openai", "anthropic", "google", "moonshot")
API_KEY = "sk-ant-NEXUSMESSAGESCANARY0123456789"
OTHER_KEY = "sk-ant-NEXUSOTHERCANARY9876543210"
KEY_ID_PATTERN = re.compile(r"^key_[0-9a-f]{16}$")
COMPLEX_PROMPT = (
    "Design a multi-region failover strategy for a payments platform: compare "
    "active-active, active-passive, and pilot-light architectures, analyze cost, "
    "latency, and consistency trade-offs, and recommend one with a phased rollout plan."
)
STREAM_HEAD = ["message_start", "content_block_start", "ping"]
STREAM_TAIL = ["content_block_stop", "message_delta", "message_stop"]


class CountingAdapter(MockProviderAdapter):
    """Mock provider that counts provider calls (``stream`` goes through ``complete``)."""

    def __init__(self, provider_name: str, *, fail: bool = False, output_tokens: int = 16) -> None:
        super().__init__(provider_name, fail=fail, output_tokens=output_tokens)
        self.calls = 0

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        self.calls += 1
        return await super().complete(model, messages, max_tokens)


class PartialStreamAdapter(MockProviderAdapter):
    """Mock provider whose stream emits one chunk and then drops the connection."""

    async def stream(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> AsyncGenerator[str, None]:
        yield f"[{model}] partial "
        raise ProviderError("connection reset mid-stream")


@dataclass
class Gateway:
    """Test client plus the router, its adapters, and the requests it dispatched."""

    client: TestClient
    router: NexusRouter
    adapters: dict[str, BaseProviderAdapter]
    ledger: SpendLedger
    dispatched: list[RouterRequest] = field(default_factory=list)

    def create(self, body: dict[str, Any], **headers: str) -> httpx.Response:
        return self.client.post("/v1/messages", json=body, headers=headers)

    def count(self, body: dict[str, Any], **headers: str) -> httpx.Response:
        return self.client.post("/v1/messages/count_tokens", json=body, headers=headers)

    @property
    def provider_calls(self) -> int:
        return sum(getattr(adapter, "calls", 0) for adapter in self.adapters.values())


GatewayFactory = Callable[..., Gateway]


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GatewayFactory:
    """Build an isolated API client backed by mock providers and tmp state files."""

    def _build(
        *,
        fail: bool = False,
        output_tokens: int = 16,
        tenant_capacity: int = 60,
        adapters: dict[str, BaseProviderAdapter] | None = None,
        **overrides: float,
    ) -> Gateway:
        settings = RouterSettings(
            audit_log_path=str(tmp_path / "audit-log.jsonl"),
            spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
            idempotency_path=str(tmp_path / "idempotency.sqlite3"),
            **overrides,
        )
        providers: dict[str, BaseProviderAdapter] = {
            name: CountingAdapter(name, fail=fail, output_tokens=output_tokens)
            for name in PROVIDERS
        }
        providers |= adapters or {}
        registry = AdapterRegistry(providers)
        router = NexusRouter(settings, registry)
        ledger = SpendLedger(settings.spend_ledger_path)
        gateway = Gateway(
            client=TestClient(main.app), router=router, adapters=providers, ledger=ledger
        )
        original_complete = router.complete
        original_stream = router.complete_stream

        async def _spy_complete(request: RouterRequest) -> RouterResponse:
            gateway.dispatched.append(request)
            return await original_complete(request)

        def _spy_stream(
            request: RouterRequest,
            *,
            on_complete: Callable[[RouterResponse], None] | None = None,
        ) -> AsyncGenerator[tuple[str, str], None]:
            gateway.dispatched.append(request)
            return original_stream(request, on_complete=on_complete)

        monkeypatch.setattr(router, "complete", _spy_complete)
        monkeypatch.setattr(router, "complete_stream", _spy_stream)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(
                capacity=tenant_capacity, refill_per_second=0.0001
            ),
        )
        cache = ResponseCache(ttl_seconds=60.0)
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: cache)
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        monkeypatch.setattr(main, "get_spend_ledger", lambda: ledger)
        return gateway

    return _build


def _body(**overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "auto",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Summarize the refund policy."}],
    }
    body.update(overrides)
    return body


def _sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in text.strip().split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            name, _, value = line.partition(": ")
            fields[name] = value
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def _streamed_text(events: list[tuple[str, dict[str, Any]]]) -> str:
    return "".join(data["delta"]["text"] for name, data in events if name == "content_block_delta")


def _assert_anthropic_error(response: httpx.Response, status: int, error_type: str) -> str:
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"type", "error"}
    assert body["type"] == "error"
    assert set(body["error"]) == {"type", "message"}
    assert body["error"]["type"] == error_type
    return str(body["error"]["message"])


def test_text_request_returns_anthropic_message_with_usage_and_routing_metadata(
    gateway: GatewayFactory,
) -> None:
    """A text request yields an Anthropic ``message`` plus the namespaced ``nexus`` block."""
    gw = gateway()

    response = gw.create(
        _body(metadata={"user_id": "user-42"}),
        **{"anthropic-version": "2026-10-21", "anthropic-beta": "fast-mode-2026-02-01"},
    )

    assert response.status_code == 200
    body = response.json()
    assert list(body) == [
        "id",
        "type",
        "role",
        "model",
        "content",
        "stop_reason",
        "stop_sequence",
        "usage",
        "nexus",
    ]
    assert re.fullmatch(r"msg_[0-9a-f]{32}", body["id"])
    assert (body["type"], body["role"]) == ("message", "assistant")
    assert body["content"] == [{"type": "text", "text": f"[{body['model']}] routed response"}]
    assert (body["stop_reason"], body["stop_sequence"]) == ("end_turn", None)
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] == 16
    (dispatched,) = gw.dispatched
    assert body["nexus"]["request_id"] == dispatched.request_id
    assert body["nexus"]["routing_strategy"] == "rule-based"
    assert body["nexus"]["cost_usd"] > 0
    assert dispatched.requested_model == "auto"
    assert dispatched.max_tokens == 256
    assert dispatched.stream is False
    assert (dispatched.user_id, dispatched.session_id) == ("user-42", "user-42")
    assert [(m.role, m.content) for m in dispatched.messages] == [
        ("user", "Summarize the refund policy.")
    ]
    assert gw.ledger.summary(tenant="user-42").request_count == 1


def test_system_blocks_and_content_blocks_map_to_router_messages(gateway: GatewayFactory) -> None:
    """System and message text blocks are joined per message; roles and order are kept."""
    gw = gateway()

    response = gw.create(
        _body(
            system=[
                {"type": "text", "text": "You are a terse support agent."},
                {
                    "type": "text",
                    "text": "Cite the policy section.",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Policy: refunds within 30 days."},
                        {"type": "text", "text": "Can I return shoes after 40 days?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "No, 30 days.", "citations": None}],
                },
                {"role": "user", "content": "And sandals?"},
            ],
        )
    )

    assert response.status_code == 200
    (dispatched,) = gw.dispatched
    assert [(m.role, m.content) for m in dispatched.messages] == [
        ("system", "You are a terse support agent.\nCite the policy section."),
        ("user", "Policy: refunds within 30 days.\nCan I return shoes after 40 days?"),
        ("assistant", "No, 30 days."),
        ("user", "And sandals?"),
    ]


@pytest.mark.parametrize(
    ("system", "content"),
    [
        ("Be terse.", "Hi there"),
        ([{"type": "text", "text": "Be terse."}], [{"type": "text", "text": "Hi there"}]),
        ("Be terse.", [{"type": "text", "text": "Hi there"}]),
        ([{"type": "text", "text": "Be terse."}], "Hi there"),
    ],
)
def test_string_and_text_block_forms_are_equivalent(
    gateway: GatewayFactory, system: object, content: object
) -> None:
    """A string is shorthand for one text block, for ``system`` and message content."""
    gw = gateway()

    response = gw.create(_body(system=system, messages=[{"role": "user", "content": content}]))

    assert response.status_code == 200
    (dispatched,) = gw.dispatched
    assert [(m.role, m.content) for m in dispatched.messages] == [
        ("system", "Be terse."),
        ("user", "Hi there"),
    ]


@pytest.mark.parametrize("max_tokens", ["missing", 0, -5, "256", 12.5, True, None])
def test_missing_or_invalid_max_tokens_is_an_anthropic_400(
    gateway: GatewayFactory, max_tokens: object
) -> None:
    """``max_tokens`` is required and must be a positive integer."""
    gw = gateway()
    body = _body()
    if max_tokens == "missing":
        del body["max_tokens"]
    else:
        body["max_tokens"] = max_tokens

    message = _assert_anthropic_error(gw.create(body), 400, "invalid_request_error")

    assert message.startswith("max_tokens: ")
    assert gw.dispatched == []


@pytest.mark.parametrize(
    ("messages", "prefix"),
    [
        ([], "messages: at least one message is required"),
        ("hello", "messages: Input should be a valid list"),
        ([{"role": "system", "content": "x"}], "messages.0.role: Input should be 'user' or"),
        ([{"role": "tool", "content": "x"}], "messages.0.role: Input should be 'user' or"),
        ([{"role": "user"}], "messages.0.content: Field required"),
        ([{"role": "user", "content": ""}], "messages.0.content: all messages must have non-"),
        ([{"role": "user", "content": []}], "messages.0.content: all messages must have non-"),
        ([{"role": "user", "content": 42}], "messages.0.content: Input should be a valid string"),
        (
            [{"role": "user", "content": [{"type": "text", "text": "  "}]}],
            "messages.0.content.0.text: text content blocks must contain non-whitespace",
        ),
        (
            [{"role": "user", "content": [{"text": "untyped"}]}],
            "messages.0.content.0.type: Field required",
        ),
        (
            [{"role": "user", "content": [{"type": "text", "text": "hi", "lang": "en"}]}],
            "messages.0.content.0.lang: Extra inputs are not permitted",
        ),
        (
            [{"role": "user", "content": "hi", "name": "alice"}],
            "messages.0.name: Extra inputs are not permitted",
        ),
        (
            [
                {"role": "user", "content": "Pick A, B or C."},
                {"role": "assistant", "content": "The answer is ("},
            ],
            "messages.1.role: the final message must have role 'user'",
        ),
    ],
)
def test_malformed_messages_are_an_anthropic_400(
    gateway: GatewayFactory, messages: object, prefix: str
) -> None:
    """Empty or malformed conversations fail before routing, naming the field."""
    gw = gateway()

    message = _assert_anthropic_error(
        gw.create(_body(messages=messages)), 400, "invalid_request_error"
    )

    assert message.startswith(prefix)
    assert gw.dispatched == []


WEATHER_TOOL = {"name": "get_weather", "input_schema": {"type": "object"}}


@pytest.mark.parametrize(
    ("overrides", "prefix"),
    [
        ({"tools": [WEATHER_TOOL]}, "tools: tool use is not supported"),
        ({"tool_choice": {"type": "any"}}, "tool_choice: only"),
        ({"tool_choice": {"type": "tool", "name": "get_weather"}}, "tool_choice: only"),
        ({"thinking": {"type": "enabled", "budget_tokens": 2048}}, "thinking: extended"),
        ({"thinking": {"type": "adaptive"}}, "thinking: extended"),
        (
            {"output_config": {"format": {"type": "json_schema", "schema": {}}}},
            "output_config.format: structured outputs are not supported",
        ),
        ({"output_config": {"effort": "extreme"}}, "output_config.effort: Input should be"),
        ({"output_config": {"verbosity": "low"}}, "output_config.verbosity: Extra inputs"),
        ({"container": "container_011"}, "container: containers"),
        ({"inference_geo": "eu"}, "inference_geo: nexus cannot pin"),
        ({"stop_sequences": ["\n"]}, "stop_sequences.0: each stop sequence must contain"),
        (
            {"mcp_servers": [{"type": "url", "url": "https://mcp.example.com"}]},
            "mcp_servers: Extra",
        ),
        ({"metadata": {"user_id": "u-1", "tenant": "acme"}}, "metadata.tenant: Extra inputs"),
        ({"n": 2}, "n: Extra inputs are not permitted"),
        (
            {"system": [{"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}]},
            "system.0.type: 'image' content blocks are not supported",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
                        ],
                    }
                ]
            },
            "messages.0.content.1.type: 'image' content blocks are not supported",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "document", "source": {"type": "text", "data": "x"}},
                        ],
                    }
                ]
            },
            "messages.0.content.0.type: 'document' content blocks are not supported",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "Weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "toolu_1", "name": "w", "input": {}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "15C"},
                        ],
                    },
                ]
            },
            "messages.1.content.0.type: 'tool_use' content blocks are not supported by nexus "
            "/v1/messages (tool use is not routed by nexus)",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "15C"},
                        ],
                    }
                ]
            },
            "messages.0.content.0.type: 'tool_result' content blocks are not supported",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "Solve it."},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "...", "signature": "sig"},
                            {"type": "text", "text": "42"},
                        ],
                    },
                    {"role": "user", "content": "Why?"},
                ]
            },
            "messages.1.content.0.type: 'thinking' content blocks are not supported",
        ),
    ],
)
def test_unsupported_features_are_rejected_naming_the_field(
    gateway: GatewayFactory, overrides: dict[str, Any], prefix: str
) -> None:
    """Tools, thinking, structured output, and non-text blocks are never silently dropped."""
    gw = gateway()

    message = _assert_anthropic_error(gw.create(_body(**overrides)), 400, "invalid_request_error")

    assert message.startswith(prefix)
    assert gw.dispatched == []
    assert gw.provider_calls == 0


@pytest.mark.parametrize("tool_choice", [{"type": "auto"}, {"type": "none"}, None])
def test_passive_and_ignored_fields_are_accepted(
    gateway: GatewayFactory, tool_choice: dict[str, str] | None
) -> None:
    """Passive defaults and documented accepted-and-ignored fields keep working."""
    gw = gateway()

    response = gw.create(
        _body(
            system=[
                {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}},
            ],
            tools=[],
            tool_choice=tool_choice,
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            stop_sequences=[],
            metadata={"user_id": None},
            service_tier="standard_only",
            speed="fast",
            cache_control={"type": "ephemeral", "ttl": "1h"},
            diagnostics={"previous_message_id": "msg_123"},
        ),
        **{"anthropic-version": "2023-06-01", "anthropic-beta": "prompt-caching-2024-07-31"},
    )

    assert response.status_code == 200
    assert response.json()["stop_reason"] == "end_turn"
    (dispatched,) = gw.dispatched
    assert [m.role for m in dispatched.messages] == ["system", "user"]


@pytest.mark.parametrize(("max_tokens", "stop_reason"), [(16, "max_tokens"), (256, "end_turn")])
def test_output_cap_maps_to_max_tokens_stop_reason(
    gateway: GatewayFactory, max_tokens: int, stop_reason: str
) -> None:
    """A provider stop at ``max_tokens`` (finish ``length``) reports ``max_tokens``."""
    gw = gateway(output_tokens=64)

    plain = gw.create(_body(max_tokens=max_tokens))
    streamed = gw.create(_body(max_tokens=max_tokens, stream=True))

    assert plain.json()["stop_reason"] == stop_reason
    assert plain.json()["usage"]["output_tokens"] == min(max_tokens, 64)
    final = _sse_events(streamed.text)[-2][1]
    assert final["delta"] == {"stop_reason": stop_reason, "stop_sequence": None}
    assert final["usage"]["output_tokens"] == min(max_tokens, 64)


def test_streaming_emits_the_anthropic_event_sequence(gateway: GatewayFactory) -> None:
    """``stream: true`` emits named events in Anthropic's order with final usage."""
    gw = gateway()

    response = gw.create(_body(stream=True), **{"x-api-key": API_KEY})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "[DONE]" not in response.text
    events = _sse_events(response.text)
    names = [name for name, _ in events]
    assert all(name == data["type"] for name, data in events)
    assert names[:3] == STREAM_HEAD
    assert names[-3:] == STREAM_TAIL
    assert set(names[3:-3]) == {"content_block_delta"}
    start = events[0][1]["message"]
    assert re.fullmatch(r"msg_[0-9a-f]{32}", start["id"])
    assert (start["type"], start["role"], start["content"]) == ("message", "assistant", [])
    assert (start["stop_reason"], start["stop_sequence"]) == (None, None)
    assert start["usage"]["output_tokens"] == 0
    assert "nexus" not in start
    assert events[1][1] == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    deltas = [data for name, data in events if name == "content_block_delta"]
    assert all(d["index"] == 0 and d["delta"]["type"] == "text_delta" for d in deltas)
    assert _streamed_text(events) == f"[{start['model']}] routed response"
    assert events[-3][1] == {"type": "content_block_stop", "index": 0}
    final = events[-2][1]
    assert final["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}
    assert final["usage"]["output_tokens"] == 16
    assert final["usage"]["input_tokens"] == start["usage"]["input_tokens"]
    (dispatched,) = gw.dispatched
    assert dispatched.stream is True
    assert final["nexus"]["request_id"] == dispatched.request_id
    assert final["nexus"]["routing_strategy"] == "rule-based"
    assert events[-1][1] == {"type": "message_stop"}


def test_streaming_records_spend_exactly_like_chat_streaming(gateway: GatewayFactory) -> None:
    """Streams use the router's accounting: one ledger row per stream, keyed by fingerprint."""
    gw = gateway()
    prompt = [{"role": "user", "content": "Summarize the refund policy."}]

    messages = gw.create(_body(stream=True, messages=prompt), **{"x-api-key": API_KEY})
    chat = gw.client.post(
        "/v1/chat/completions",
        json={"model": "auto", "max_tokens": 256, "messages": prompt, "stream": True},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )

    assert messages.status_code == chat.status_code == 200
    summary = gw.ledger.summary()
    assert summary.request_count == 2
    ((tenant, cost),) = summary.by_tenant.items()
    assert tenant == fingerprint_api_key(API_KEY)
    message_cost = _sse_events(messages.text)[-2][1]["nexus"]["cost_usd"]
    assert cost == pytest.approx(2 * message_cost)


def test_stop_sequences_cut_the_output_and_report_the_match(gateway: GatewayFactory) -> None:
    """``stop_sequences`` are enforced at the gateway with ``stop_reason: stop_sequence``."""
    gw = gateway()

    cut = gw.create(_body(stop_sequences=["zzz", "routed"])).json()
    uncut = gw.create(_body(stop_sequences=["zzz"])).json()

    assert cut["content"] == [{"type": "text", "text": f"[{cut['model']}] "}]
    assert (cut["stop_reason"], cut["stop_sequence"]) == ("stop_sequence", "routed")
    assert cut["usage"]["output_tokens"] == 16
    assert uncut["content"][0]["text"] == f"[{uncut['model']}] routed response"
    assert (uncut["stop_reason"], uncut["stop_sequence"]) == ("end_turn", None)


def test_streamed_stop_sequences_never_leak_cut_text(gateway: GatewayFactory) -> None:
    """Text after a stop sequence is never streamed, and the stream is still accounted."""
    gw = gateway()

    response = gw.create(_body(stream=True, stop_sequences=["routed"]))

    events = _sse_events(response.text)
    start = events[0][1]["message"]
    assert _streamed_text(events) == f"[{start['model']}] "
    final = events[-2][1]
    assert final["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "routed"}
    assert final["usage"]["output_tokens"] == 16
    assert [name for name, _ in events][-3:] == STREAM_TAIL
    assert gw.ledger.summary().request_count == 1


def test_stop_sequence_filter_handles_chunk_boundaries_and_earliest_end() -> None:
    """Held-back text spans chunks, and the sequence that ends first wins."""
    split = StopSequenceFilter(["STOP"])
    assert [split.feed("Hello ST"), split.feed("OP and more"), split.flush()] == [
        "Hello",
        " ",
        "",
    ]
    assert split.matched == "STOP"

    no_match = StopSequenceFilter(["STOP"])
    pieces = [no_match.feed("Hello ST"), no_match.feed("AR"), no_match.flush()]
    assert "".join(pieces) == "Hello STAR"
    assert no_match.matched is None

    earliest = StopSequenceFilter(["abcd", "bc"])
    assert "".join([earliest.feed("xa"), earliest.feed("b"), earliest.feed("cdy")]) == "xa"
    assert earliest.matched == "bc"

    assert apply_stop_sequences("no stop here", []) == ("no stop here", None)
    assert apply_stop_sequences("a</answer>b", ["</answer>"]) == ("a", "</answer>")


@pytest.mark.parametrize(
    ("exception", "error_type"),
    [
        (RoutingFailedError("all provider attempts failed: upstream reset"), "overloaded_error"),
        (
            RateLimitExceededError("rate limit exceeded for key_0123456789abcdef"),
            "rate_limit_error",
        ),
        (BudgetExceededError("budget cap exceeded for key_0123456789abcdef"), "billing_error"),
    ],
)
def test_stream_failure_after_output_ends_with_an_error_event(
    gateway: GatewayFactory,
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    error_type: str,
) -> None:
    """A routing failure after text was streamed ends the stream with ``event: error``."""
    gw = gateway()

    async def _partial_stream(
        request: RouterRequest, **_: object
    ) -> AsyncGenerator[tuple[str, str], None]:
        yield "gpt-6-luna", "partial "
        raise exception

    monkeypatch.setattr(gw.router, "complete_stream", _partial_stream)

    response = gw.create(_body(stream=True))

    assert response.status_code == 200
    events = _sse_events(response.text)
    names = [name for name, _ in events]
    assert names == [*STREAM_HEAD, "content_block_delta", "error"]
    assert events[0][1]["message"]["model"] == "gpt-6-luna"
    assert events[-1][1] == {
        "type": "error",
        "error": {"type": error_type, "message": str(exception)},
    }


def test_stream_does_not_splice_a_fallback_answer_after_text(gateway: GatewayFactory) -> None:
    """A provider failing mid-stream (real router) ends with an error, never a second answer."""
    gw = gateway(adapters={"google": PartialStreamAdapter("google")})

    response = gw.create(_body(stream=True))

    events = _sse_events(response.text)
    assert events[0][1]["message"]["model"] == GEMINI_FLASH_MODEL
    assert _streamed_text(events) == f"[{GEMINI_FLASH_MODEL}] partial "
    name, error = events[-1]
    assert name == "error"
    assert error["error"]["type"] == "overloaded_error"
    assert "after streaming output began" in error["error"]["message"]
    assert "message_stop" not in [name for name, _ in events]


def test_stream_failure_before_output_is_a_real_http_error(gateway: GatewayFactory) -> None:
    """When every provider fails before any text, streams get an HTTP 503, not a 200."""
    gw = gateway(fail=True)

    response = gw.create(_body(stream=True))

    message = _assert_anthropic_error(response, 503, "overloaded_error")
    assert message.startswith("all provider attempts failed")


def test_count_tokens_returns_the_router_estimate_without_a_provider_call(
    gateway: GatewayFactory,
) -> None:
    """``count_tokens`` is the routing/budget prompt estimate and never calls a provider."""
    gw = gateway()
    body = {
        "model": ANTHROPIC_SAFETY_MODEL,
        "system": [{"type": "text", "text": "You are a terse support agent."}],
        "messages": [{"role": "user", "content": COMPLEX_PROMPT}],
    }

    counted = gw.count(body, **{"x-api-key": API_KEY, "anthropic-version": "2026-10-21"})

    assert counted.status_code == 200
    assert list(counted.json()) == ["input_tokens"]
    input_tokens = counted.json()["input_tokens"]
    assert input_tokens == len(f"You are a terse support agent.\n{COMPLEX_PROMPT}") // 4
    assert gw.provider_calls == 0
    assert gw.dispatched == []
    assert gw.ledger.summary().request_count == 0

    streamed = gw.create({**body, "max_tokens": 64, "stream": True})

    (dispatched,) = gw.dispatched
    assert RequestAnalyzer().analyze(dispatched).prompt_tokens_estimate == input_tokens
    assert _sse_events(streamed.text)[0][1]["message"]["usage"]["input_tokens"] == input_tokens


@pytest.mark.parametrize(
    ("body", "status", "error_type", "prefix"),
    [
        (
            {"model": "auto", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 9},
            400,
            "invalid_request_error",
            "max_tokens: Extra inputs are not permitted",
        ),
        (
            {"messages": [{"role": "user", "content": "hi"}]},
            400,
            "invalid_request_error",
            "model: Field required",
        ),
        (
            {"model": "auto", "messages": [{"role": "user", "content": "hi"}], "tools": [{}]},
            400,
            "invalid_request_error",
            "tools: tool use is not supported",
        ),
        (
            {
                "model": "auto",
                "messages": [
                    {"role": "user", "content": [{"type": "image", "source": {"type": "url"}}]}
                ],
            },
            400,
            "invalid_request_error",
            "messages.0.content.0.type: 'image' content blocks are not supported",
        ),
        (
            {"model": "auto", "messages": []},
            400,
            "invalid_request_error",
            "messages: at least one message is required",
        ),
    ],
)
def test_count_tokens_validates_like_messages(
    gateway: GatewayFactory, body: dict[str, Any], status: int, error_type: str, prefix: str
) -> None:
    """``count_tokens`` shares the Messages validation and the Anthropic error envelope."""
    gw = gateway()

    message = _assert_anthropic_error(gw.count(body), status, error_type)

    assert message.startswith(prefix)
    assert gw.provider_calls == 0


def test_x_api_key_and_bearer_token_share_one_identity(gateway: GatewayFactory) -> None:
    """``x-api-key`` maps to the same fingerprint, bucket, and spend tenant as ``Bearer``."""
    gw = gateway(rate_limit_capacity=1, rate_limit_refill_per_second=0.0001)

    first = gw.create(_body(), **{"x-api-key": API_KEY})
    limited = gw.create(
        _body(messages=[{"role": "user", "content": "And for gift cards?"}]),
        Authorization=f"Bearer {API_KEY}",
    )
    other = gw.create(_body(), **{"x-api-key": OTHER_KEY})

    assert first.status_code == other.status_code == 200
    message = _assert_anthropic_error(limited, 429, "rate_limit_error")
    assert API_KEY not in message
    assert fingerprint_api_key(API_KEY) in message
    first_request, limited_request, other_request = gw.dispatched
    assert first_request.api_key_id == limited_request.api_key_id
    assert first_request.api_key_id == fingerprint_api_key(f"Bearer {API_KEY}")
    assert KEY_ID_PATTERN.match(first_request.api_key_id)
    assert other_request.api_key_id != first_request.api_key_id
    for request in gw.dispatched:
        assert API_KEY not in request.model_dump_json()
    assert API_KEY not in gw.client.get("/v1/spend").text


def test_conflicting_credentials_are_a_401_that_never_echoes_keys(gateway: GatewayFactory) -> None:
    """Different keys in ``x-api-key`` and ``Authorization`` are rejected; equal keys are fine."""
    gw = gateway()
    conflicting = {"x-api-key": API_KEY, "Authorization": f"Bearer {OTHER_KEY}"}

    rejected = gw.create(_body(), **conflicting)
    rejected_count = gw.count(
        {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}, **conflicting
    )
    same_key = gw.create(_body(), **{"x-api-key": API_KEY, "Authorization": f"Bearer {API_KEY}"})

    for response in (rejected, rejected_count):
        message = _assert_anthropic_error(response, 401, "authentication_error")
        assert API_KEY not in message
        assert OTHER_KEY not in message
    assert same_key.status_code == 200
    (dispatched,) = gw.dispatched
    assert dispatched.api_key_id == fingerprint_api_key(API_KEY)


def test_requests_without_credentials_route_anonymously_like_chat(
    gateway: GatewayFactory,
) -> None:
    """Nexus does not enforce API keys, so a missing or empty key routes as ``anonymous``."""
    gw = gateway()

    missing = gw.create(_body())
    empty = gw.create(_body(max_tokens=128), **{"x-api-key": "", "Authorization": "Bearer "})
    bearer = gw.create(_body(max_tokens=64), **{"x-api-key": "", "Authorization": API_KEY})

    assert missing.status_code == empty.status_code == bearer.status_code == 200
    assert [request.api_key_id for request in gw.dispatched] == [
        "anonymous",
        "anonymous",
        fingerprint_api_key(API_KEY),
    ]


def test_tenant_rate_limit_is_an_anthropic_429(gateway: GatewayFactory) -> None:
    """``X-Tenant-Id`` applies the hard tenant bucket before routing."""
    gw = gateway(tenant_capacity=1)

    assert gw.create(_body(), **{"X-Tenant-Id": "acme"}).status_code == 200
    limited = gw.create(_body(max_tokens=64), **{"X-Tenant-Id": "acme"})

    _assert_anthropic_error(limited, 429, "rate_limit_error")
    assert len(gw.dispatched) == 1


@pytest.mark.parametrize(
    ("exception", "status", "error_type"),
    [
        (
            RateLimitExceededError("rate limit exceeded for key_0123456789abcdef"),
            429,
            "rate_limit_error",
        ),
        (BudgetExceededError("budget cap exceeded for key_0123456789abcdef"), 402, "billing_error"),
        (RoutingFailedError("all provider attempts failed: boom"), 503, "overloaded_error"),
    ],
)
def test_router_errors_map_to_anthropic_status_and_type(
    gateway: GatewayFactory,
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    status: int,
    error_type: str,
) -> None:
    """Nexus keeps its 429 / 402 / 503 statuses and uses the matching Anthropic types."""
    gw = gateway()

    async def _raise(request: RouterRequest) -> RouterResponse:
        raise exception

    monkeypatch.setattr(gw.router, "complete", _raise)

    assert _assert_anthropic_error(gw.create(_body()), status, error_type) == str(exception)


def test_budget_exhaustion_is_a_routing_failure_that_hides_the_key(
    gateway: GatewayFactory,
) -> None:
    """A spent budget skips every candidate, so the router reports 503 like the other APIs."""
    gw = gateway(budget_cap_usd=0.0)

    response = gw.create(_body(), **{"x-api-key": API_KEY})

    message = _assert_anthropic_error(response, 503, "overloaded_error")
    assert "budget cap exceeded" in message
    assert API_KEY not in message


def test_idempotency_key_replays_and_is_scoped_per_endpoint(gateway: GatewayFactory) -> None:
    """A retry replays the stored message; the same key on other endpoints never collides."""
    gw = gateway()
    key = {"Idempotency-Key": "order-42"}

    first = gw.create(_body(), **key)
    retry = gw.create(_body(), **key)
    responses = gw.client.post(
        "/v1/responses", json={"input": "Same key, Responses API."}, headers=key
    )
    chat = gw.client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Same key, Chat Completions."}]},
        headers=key,
    )
    replay_after = gw.create(_body(), **key)
    streamed = gw.create(_body(stream=True), **key)

    assert first.status_code == retry.status_code == 200
    assert retry.json() == first.json()
    assert responses.json()["object"] == "response"
    assert chat.json()["object"] == "chat.completion"
    assert replay_after.json() == first.json()
    # Streams are never stored or replayed, as on /v1/responses.
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert len(gw.dispatched) == 4


def test_idempotency_key_in_flight_is_a_409_conflict_error(gateway: GatewayFactory) -> None:
    """A duplicate of a still-running request gets 409 instead of a second provider call."""
    gw = gateway()
    main.get_gateway_guards().reserve_idempotency("order-7\n/v1/messages", "anonymous")

    duplicate = gw.create(_body(), **{"Idempotency-Key": "order-7"})
    chat = gw.client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Same raw key."}]},
        headers={"Idempotency-Key": "order-7"},
    )

    message = _assert_anthropic_error(duplicate, 409, "conflict_error")
    assert "still in progress" in message
    assert chat.status_code == 200
    assert len(gw.dispatched) == 1


def test_failed_request_releases_its_idempotency_key(
    gateway: GatewayFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure is not stored, so retrying the same key routes again."""
    gw = gateway()
    key = {"Idempotency-Key": "order-9"}
    spy = gw.router.complete

    async def _fail(request: RouterRequest) -> RouterResponse:
        raise RoutingFailedError("all provider attempts failed: boom")

    monkeypatch.setattr(gw.router, "complete", _fail)
    failed = gw.create(_body(), **key)
    monkeypatch.setattr(gw.router, "complete", spy)
    retried = gw.create(_body(), **key)

    _assert_anthropic_error(failed, 503, "overloaded_error")
    assert retried.status_code == 200
    assert retried.json()["type"] == "message"


def test_identical_requests_share_the_response_cache(gateway: GatewayFactory) -> None:
    """Non-streaming requests reuse the exact-match cache shared with Chat Completions."""
    gw = gateway()

    first = gw.create(_body())
    second = gw.create(_body())
    chat = gw.client.post(
        "/v1/chat/completions",
        json={"model": "auto", "max_tokens": 256, "messages": _body()["messages"]},
    )
    other_limit = gw.create(_body(max_tokens=64))

    assert first.json()["id"] != second.json()["id"]
    assert first.json()["nexus"] == second.json()["nexus"]
    assert chat.json()["router"]["request_id"] == first.json()["nexus"]["request_id"]
    assert other_limit.status_code == 200
    assert [request.max_tokens for request in gw.dispatched] == [256, 64]


def test_strategy_header_routes_and_unknown_strategy_is_a_400(gateway: GatewayFactory) -> None:
    """``X-Router-Strategy`` picks the strategy; unknown names are rejected before routing."""
    gw = gateway()

    routed = gw.create(_body(), **{"X-Router-Strategy": "cost-optimal"})
    invalid = gw.create(_body(), **{"X-Router-Strategy": "nope"})

    assert routed.json()["nexus"]["routing_strategy"] == "cost-optimal"
    message = _assert_anthropic_error(invalid, 400, "invalid_request_error")
    assert message == "unsupported routing strategy: nope"
    assert len(gw.dispatched) == 1


@pytest.mark.parametrize(
    ("requested", "served"),
    [(ANTHROPIC_SAFETY_MODEL, ANTHROPIC_SAFETY_MODEL), ("gpt-6-astra", "gpt-6-astra")],
)
def test_explicit_catalog_model_is_honored_like_chat_completions(
    gateway: GatewayFactory, requested: str, served: str
) -> None:
    """A catalog model id pins the request when the strategy honors explicit models."""
    gw = gateway()

    body = gw.create(_body(model=requested, messages=[{"role": "user", "content": COMPLEX_PROMPT}]))

    assert body.json()["model"] == served
    assert body.json()["nexus"]["rationale"] == "explicit compatible model request honored"


def test_unknown_messages_subpaths_and_methods_use_the_anthropic_envelope(
    gateway: GatewayFactory,
) -> None:
    """404/405 below ``/v1/messages`` are Anthropic errors; other paths keep FastAPI's."""
    gw = gateway()

    batches = gw.client.post("/v1/messages/batches", json={"requests": []})
    wrong_method = gw.client.get("/v1/messages")
    elsewhere = gw.client.get("/v1/not-a-route")

    _assert_anthropic_error(batches, 404, "not_found_error")
    _assert_anthropic_error(wrong_method, 405, "invalid_request_error")
    assert wrong_method.headers["allow"] == "POST"
    assert elsewhere.status_code == 404
    assert elsewhere.json() == {"detail": "Not Found"}


def test_invalid_json_body_is_an_anthropic_400(gateway: GatewayFactory) -> None:
    """A body that is not JSON gets the Anthropic envelope too."""
    gw = gateway()

    response = gw.client.post(
        "/v1/messages", content=b"{not json", headers={"content-type": "application/json"}
    )

    message = _assert_anthropic_error(response, 400, "invalid_request_error")
    assert message == "The request body is not valid JSON."


def test_other_endpoints_keep_their_openai_and_fastapi_envelopes(
    gateway: GatewayFactory,
) -> None:
    """Regression guard: only ``/v1/messages*`` switched to the Anthropic envelope."""
    gw = gateway(rate_limit_capacity=1, rate_limit_refill_per_second=0.0001)
    chat_body = {"messages": [{"role": "user", "content": "hi"}]}
    other_body = {"messages": [{"role": "user", "content": "hello again"}]}

    chat_invalid = gw.client.post("/v1/chat/completions", json={"messages": "not-a-list"})
    responses_invalid = gw.client.post("/v1/responses", json={"input": []})
    chat_ok = gw.client.post("/v1/chat/completions", json=chat_body, headers={"x-api-key": API_KEY})
    chat_limited = gw.client.post(
        "/v1/chat/completions", json=other_body, headers={"x-api-key": OTHER_KEY}
    )

    assert chat_invalid.status_code == 422
    assert "detail" in chat_invalid.json()
    assert responses_invalid.status_code == 400
    assert set(responses_invalid.json()["error"]) == {"message", "type", "param", "code"}
    assert chat_ok.status_code == 200
    # Chat Completions identifies callers by ``Authorization`` only, so both
    # x-api-key requests share the anonymous bucket and keep the FastAPI body.
    assert chat_limited.status_code == 429
    assert set(chat_limited.json()) == {"detail"}
    assert [request.api_key_id for request in gw.dispatched] == ["anonymous", "anonymous"]
