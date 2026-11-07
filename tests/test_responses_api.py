"""Tests for the OpenAI Responses API compatible ``POST /v1/responses`` endpoint."""

from __future__ import annotations

import json
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
from cache.response_cache import ResponseCache
from router.config import RouterSettings
from router.engine import NexusRouter, RoutingFailedError
from router.schemas import ChatMessage, RouterRequest, RouterResponse
from safety.budget import BudgetExceededError
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.rate_limiter import RateLimitExceededError
from safety.tenant_rate_limiter import TenantRateLimiter

PROVIDERS = ("openai", "anthropic", "google", "moonshot")
STREAM_ORDER_HEAD = [
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.content_part.added",
]
STREAM_ORDER_TAIL = [
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.completed",
]


class PartialStreamAdapter(MockProviderAdapter):
    """Mock provider whose stream emits one chunk and then drops the connection."""

    async def stream(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> AsyncGenerator[str, None]:
        yield f"[{model}] partial "
        raise ProviderError("connection reset mid-stream")


class EmptyThenFailAdapter(MockProviderAdapter):
    """Mock provider that emits an empty chunk and then fails before any text."""

    async def stream(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> AsyncGenerator[str, None]:
        yield ""
        raise ProviderError("connection reset before first token")


class EmptyAnswerAdapter(MockProviderAdapter):
    """Mock provider that successfully streams an empty answer."""

    async def stream(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> AsyncGenerator[str, None]:
        yield ""


class SilentStreamAdapter(MockProviderAdapter):
    """Mock provider whose successful stream yields no chunks at all."""

    async def stream(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> AsyncGenerator[str, None]:
        for _ in ():
            yield ""


@dataclass
class Gateway:
    """Test client plus the router and the requests it dispatched."""

    client: TestClient
    router: NexusRouter
    dispatched: list[RouterRequest] = field(default_factory=list)

    def create(self, body: dict[str, Any], **headers: str) -> httpx.Response:
        return self.client.post("/v1/responses", json=body, headers=headers)


GatewayFactory = Callable[..., Gateway]


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GatewayFactory:
    """Build an isolated API client backed by mock providers and tmp state files."""

    def _build(
        *,
        fail: bool = False,
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
        registry = AdapterRegistry(
            {name: MockProviderAdapter(name, fail=fail) for name in PROVIDERS} | (adapters or {})
        )
        router = NexusRouter(settings, registry)
        gateway = Gateway(client=TestClient(main.app), router=router)
        original_complete = router.complete

        async def _spy(request: RouterRequest) -> RouterResponse:
            gateway.dispatched.append(request)
            return await original_complete(request)

        monkeypatch.setattr(router, "complete", _spy)
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
        return gateway

    return _build


def _sse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.strip().split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            name, _, value = line.partition(": ")
            fields[name] = value
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def _assert_openai_error(response: httpx.Response, status: int, code: str | None) -> dict[str, Any]:
    assert response.status_code == status
    error = response.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["code"] == code
    return dict(error)


def test_string_input_returns_completed_response_object(gateway: GatewayFactory) -> None:
    """A plain string input yields an OpenAI ``response`` object with nexus metadata."""
    gw = gateway()

    response = gw.create(
        {"model": "auto", "input": "Summarize the refund policy.", "metadata": {"ticket": "T-1"}}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("resp_")
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["error"] is None
    assert body["incomplete_details"] is None
    assert body["metadata"] == {"ticket": "T-1"}
    assert body["store"] is False
    (message,) = body["output"]
    assert message["type"] == "message"
    assert message["role"] == "assistant"
    assert message["status"] == "completed"
    assert message["id"].startswith("msg_")
    (part,) = message["content"]
    assert part == {
        "type": "output_text",
        "text": f"[{body['model']}] routed response",
        "annotations": [],
        "logprobs": [],
    }
    usage = body["usage"]
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["input_tokens_details"] == {"cached_tokens": 0}
    assert usage["output_tokens_details"] == {"reasoning_tokens": 0}
    (dispatched,) = gw.dispatched
    assert body["nexus"]["request_id"] == dispatched.request_id
    assert body["nexus"]["routing_strategy"] == "rule-based"
    assert body["nexus"]["cost_usd"] > 0
    assert dispatched.requested_model == "auto"
    assert [(m.role, m.content) for m in dispatched.messages] == [
        ("user", "Summarize the refund policy.")
    ]


def test_message_items_parts_and_instructions_map_to_router_messages(
    gateway: GatewayFactory,
) -> None:
    """Instructions, developer/assistant roles, and text parts translate faithfully."""
    gw = gateway()

    response = gw.create(
        {
            "instructions": "Answer in one sentence.",
            "max_output_tokens": 64,
            "temperature": 0.2,
            "input": [
                {"role": "developer", "content": "Cite the policy section."},
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Policy: refunds within 30 days."},
                        {"type": "input_text", "text": "Can I return shoes after 40 days?"},
                    ],
                },
                {
                    "type": "message",
                    "id": "msg_prev",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Earlier answer.", "annotations": []}
                    ],
                },
                {"role": "user", "content": "And sandals?"},
            ],
        }
    )

    assert response.status_code == 200
    body = response.json()
    assert body["instructions"] == "Answer in one sentence."
    assert body["max_output_tokens"] == 64
    assert body["temperature"] == 0.2
    (dispatched,) = gw.dispatched
    assert dispatched.max_tokens == 64
    assert [(m.role, m.content) for m in dispatched.messages] == [
        ("system", "Answer in one sentence."),
        ("system", "Cite the policy section."),
        ("user", "Policy: refunds within 30 days.\nCan I return shoes after 40 days?"),
        ("assistant", "Earlier answer."),
        ("user", "And sandals?"),
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("previous_response_id", "resp_123"),
        ("conversation", "conv_123"),
        ("background", True),
        ("store", True),
        ("tools", [{"type": "function", "name": "lookup", "parameters": {}}]),
        ("tool_choice", "required"),
        ("prompt", {"id": "pmpt_123"}),
        ("text", {"format": {"type": "json_schema", "name": "x", "schema": {}}}),
        ("text", {"format": {"type": "json_object"}}),
    ],
)
def test_unsupported_features_are_rejected_with_openai_400(
    gateway: GatewayFactory, field: str, value: object
) -> None:
    """Features nexus cannot honor fail loudly instead of being silently dropped."""
    gw = gateway()

    response = gw.create({"input": "hi", field: value})

    error = _assert_openai_error(response, 400, "unsupported_parameter")
    assert error["type"] == "invalid_request_error"
    assert error["param"] == ("text.format" if field == "text" else field)
    assert gw.dispatched == []


@pytest.mark.parametrize(
    ("text", "param"),
    [
        ("plain", "text"),
        ({"format": "json_object"}, "text.format"),
        ({"format": {}}, "text.format.type"),
        ({"format": {"type": "yaml"}}, "text.format.type"),
        ({"format": {"type": {"nested": "text"}}}, "text.format.type"),
    ],
)
def test_malformed_text_format_is_rejected_as_invalid_value(
    gateway: GatewayFactory, text: object, param: str
) -> None:
    """Malformed ``text``/``text.format`` is a 400 instead of a silent plain-text answer."""
    gw = gateway()

    error = _assert_openai_error(gw.create({"input": "hi", "text": text}), 400, "invalid_value")

    assert error["param"] == param
    assert gw.dispatched == []


def test_passive_defaults_of_unsupported_features_are_accepted(gateway: GatewayFactory) -> None:
    """``store: false``, empty tools, and passive tool_choice keep working."""
    gw = gateway()

    response = gw.create(
        {
            "input": "hi",
            "store": False,
            "background": False,
            "tools": [],
            "tool_choice": "none",
            "text": {"format": {"type": "text"}},
            "top_p": 0.9,
            "unknown_vendor_field": {"ignored": True},
        }
    )

    assert response.status_code == 200
    assert response.json()["tool_choice"] == "none"


@pytest.mark.parametrize(
    ("item", "param"),
    [
        (
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image?"},
                    {"type": "input_image", "image_url": "https://example.com/cat.png"},
                ],
            },
            "input[0].content[1].type",
        ),
        ({"type": "function_call_output", "call_id": "c1", "output": "{}"}, "input[0].type"),
        ({"role": "tool", "content": "x"}, "input[0].role"),
    ],
)
def test_non_text_input_items_are_rejected(
    gateway: GatewayFactory, item: dict[str, Any], param: str
) -> None:
    """Non-text parts and non-message items get a precise 400 instead of a lossy route."""
    gw = gateway()

    response = gw.create({"input": [item]})

    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    assert gw.dispatched == []


@pytest.mark.parametrize(
    ("body", "param", "code"),
    [
        ({"model": "auto"}, "input", "missing_required_parameter"),
        ({"input": []}, "input", "empty_array"),
        ({"input": "hi", "max_output_tokens": 8}, "max_output_tokens", "invalid_value"),
        ({"input": "hi", "temperature": 2.5}, "temperature", "invalid_value"),
        (
            {"input": "hi", "metadata": {str(i): "v" for i in range(17)}},
            "metadata",
            "invalid_value",
        ),
    ],
)
def test_validation_errors_use_the_openai_error_envelope(
    gateway: GatewayFactory, body: dict[str, Any], param: str, code: str
) -> None:
    """Request validation failures on /v1/responses are OpenAI-style 400s."""
    gw = gateway()

    error = _assert_openai_error(gw.create(body), 400, code)

    assert error["param"] == param


def test_chat_completions_keeps_fastapi_validation_errors(gateway: GatewayFactory) -> None:
    """The scoped validation handler must not change other routes."""
    gw = gateway()

    response = gw.client.post("/v1/chat/completions", json={"messages": "not-a-list"})

    assert response.status_code == 422
    assert "detail" in response.json()


def test_streaming_emits_official_event_order_with_sequence_numbers(
    gateway: GatewayFactory,
) -> None:
    """``stream: true`` emits the documented SSE lifecycle ending in ``response.completed``."""
    gw = gateway()

    response = gw.create({"input": "Stream a status update.", "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "[DONE]" not in response.text
    events = _sse_events(response.text)
    names = [name for name, _ in events]
    assert all(name == data["type"] for name, data in events)
    assert names[:4] == STREAM_ORDER_HEAD
    assert names[-4:] == STREAM_ORDER_TAIL
    assert set(names[4:-4]) == {"response.output_text.delta"}
    assert [data["sequence_number"] for _, data in events] == list(range(len(events)))

    created = events[0][1]["response"]
    completed = events[-1][1]["response"]
    assert created["status"] == "in_progress"
    assert created["usage"] is None
    assert completed["id"] == created["id"] == events[1][1]["response"]["id"]
    item_id = events[2][1]["item"]["id"]
    deltas = [data for name, data in events if name == "response.output_text.delta"]
    assert all(d["item_id"] == item_id and d["output_index"] == 0 for d in deltas)
    assert all(d["content_index"] == 0 and d["logprobs"] == [] for d in deltas)
    streamed_text = "".join(d["delta"] for d in deltas)
    assert events[-4][1]["text"] == streamed_text
    assert completed["status"] == "completed"
    assert completed["output"][0]["id"] == item_id
    assert completed["output"][0]["content"][0]["text"] == streamed_text
    usage = completed["usage"]
    assert usage["output_tokens"] > 0
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert completed["nexus"]["routing_strategy"] == "rule-based"
    assert completed["model"] == created["model"]
    assert streamed_text == f"[{completed['model']}] routed response"


def test_stream_failure_after_output_ends_with_response_failed(
    gateway: GatewayFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A routing failure after deltas were sent ends the stream with ``response.failed``."""
    gw = gateway()

    async def _partial_stream(
        request: RouterRequest, **_: object
    ) -> AsyncGenerator[tuple[str, str], None]:
        yield "gpt-mock", "partial "
        raise RoutingFailedError("all provider attempts failed: upstream reset")

    monkeypatch.setattr(gw.router, "complete_stream", _partial_stream)

    response = gw.create({"input": "hi", "stream": True})

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert [name for name, _ in events][-2:] == ["response.output_text.delta", "response.failed"]
    failed = events[-1][1]["response"]
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "server_error"
    assert failed["error"]["type"] == "routing_failed"
    assert "upstream reset" in failed["error"]["message"]
    assert failed["output"][0]["status"] == "incomplete"
    assert failed["output"][0]["content"][0]["text"] == "partial "
    assert failed["id"] == events[0][1]["response"]["id"]


def test_stream_failure_before_output_returns_http_error(gateway: GatewayFactory) -> None:
    """When routing fails before any output, streaming requests get a real HTTP status."""
    gw = gateway(fail=True)

    response = gw.create({"input": "hi", "stream": True})

    error = _assert_openai_error(response, 503, "routing_failed")
    assert error["type"] == "routing_failed"


@pytest.mark.parametrize(
    ("exception", "status", "code"),
    [
        (
            RateLimitExceededError("rate limit exceeded for key_0123456789abcdef"),
            429,
            "rate_limit_exceeded",
        ),
        (
            BudgetExceededError("budget cap exceeded for key_0123456789abcdef"),
            402,
            "budget_exceeded",
        ),
        (RoutingFailedError("all provider attempts failed: boom"), 503, "routing_failed"),
    ],
)
def test_router_errors_map_to_http_status_and_openai_envelope(
    gateway: GatewayFactory,
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    status: int,
    code: str,
) -> None:
    """429 / 402 / 503 mirror /v1/chat/completions with OpenAI-style bodies."""
    gw = gateway()

    async def _raise(request: RouterRequest) -> RouterResponse:
        raise exception

    monkeypatch.setattr(gw.router, "complete", _raise)

    error = _assert_openai_error(gw.create({"input": "hi"}), status, code)
    assert error["message"] == str(exception)


def test_strategy_header_and_api_key_fingerprint_are_applied(gateway: GatewayFactory) -> None:
    """X-Router-Strategy routes the request; the raw Authorization header never leaks."""
    gw = gateway(rate_limit_capacity=1, rate_limit_refill_per_second=0.0001)
    headers = {
        "Authorization": "Bearer sk-live-RESPONSESCANARY42",
        "X-Router-Strategy": "cost-optimal",
    }

    first = gw.create({"input": "Summarize."}, **headers)
    limited = gw.create({"input": "Summarize again."}, **headers)
    invalid = gw.create({"input": "hi"}, **{"X-Router-Strategy": "nope"})

    assert first.json()["nexus"]["routing_strategy"] == "cost-optimal"
    assert gw.dispatched[0].api_key_id.startswith("key_")
    error = _assert_openai_error(limited, 429, "rate_limit_exceeded")
    assert "RESPONSESCANARY42" not in error["message"]
    assert _assert_openai_error(invalid, 400, "invalid_value")["param"] == "X-Router-Strategy"


def test_tenant_rate_limit_rejects_before_routing(gateway: GatewayFactory) -> None:
    """X-Tenant-Id applies the hard tenant bucket with an OpenAI-style 429."""
    gw = gateway(tenant_capacity=1)

    assert gw.create({"input": "one"}, **{"X-Tenant-Id": "acme"}).status_code == 200
    limited = gw.create({"input": "two"}, **{"X-Tenant-Id": "acme"})

    error = _assert_openai_error(limited, 429, "rate_limit_exceeded")
    assert error["type"] == "rate_limit_error"
    assert len(gw.dispatched) == 1


def test_idempotency_key_replays_non_streaming_response(gateway: GatewayFactory) -> None:
    """Retries with the same Idempotency-Key return the stored response verbatim."""
    gw = gateway()
    key = {"Idempotency-Key": "order-42"}

    first = gw.create({"input": "Create the refund note."}, **key)
    retry = gw.create({"input": "Create the refund note."}, **key)
    chat = gw.client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Same key, other endpoint."}]},
        headers=key,
    )

    prefixed_chat = gw.client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Prefixed key, other endpoint."}]},
        headers={"Idempotency-Key": "/v1/responses:order-42"},
    )

    assert first.status_code == retry.status_code == 200
    assert retry.json() == first.json()
    # Keys are scoped per endpoint, so neither chat call is served the stored body,
    # not even one whose raw key spells out the Responses scope.
    assert chat.json()["object"] == "chat.completion"
    assert prefixed_chat.json()["object"] == "chat.completion"
    assert len(gw.dispatched) == 3


def test_identical_requests_share_the_response_cache(gateway: GatewayFactory) -> None:
    """Non-streaming requests reuse the exact-match cache keyed on all generation inputs."""
    gw = gateway()
    body = {"input": "Cache me.", "max_output_tokens": 32}

    first = gw.create(body)
    second = gw.create(body)
    other_limit = gw.create({**body, "max_output_tokens": 64})

    assert first.json()["id"] != second.json()["id"]
    assert first.json()["nexus"] == second.json()["nexus"]
    assert [request.max_tokens for request in gw.dispatched] == [32, 64]
    assert other_limit.status_code == 200


def test_stream_does_not_splice_fallback_output_after_text_was_sent(
    gateway: GatewayFactory,
) -> None:
    """Once text streamed, a provider failure must not splice a fallback answer on."""
    gw = gateway(adapters={"google": PartialStreamAdapter("google")})

    response = gw.create({"input": "hi", "stream": True})

    events = _sse_events(response.text)
    names = [name for name, _ in events]
    assert names[-1] == "response.failed"
    assert "response.completed" not in names
    deltas = [data["delta"] for name, data in events if name == "response.output_text.delta"]
    failed = events[-1][1]["response"]
    assert deltas == ["[gemini-3.5-flash] partial "]
    assert failed["output"][0]["content"][0]["text"] == "[gemini-3.5-flash] partial "
    assert failed["error"]["type"] == "routing_failed"
    assert "after streaming output began" in failed["error"]["message"]


def test_stream_preflight_skips_empty_chunks_before_failing(
    gateway: GatewayFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty chunks are not output: a later routing failure is still an HTTP error."""
    gw = gateway()

    async def _empty_then_fail(
        request: RouterRequest, **_: object
    ) -> AsyncGenerator[tuple[str, str], None]:
        yield "gpt-mock", ""
        raise RoutingFailedError("all provider attempts failed: boom")

    monkeypatch.setattr(gw.router, "complete_stream", _empty_then_fail)

    response = gw.create({"input": "hi", "stream": True})

    assert _assert_openai_error(response, 503, "routing_failed")["type"] == "routing_failed"


def test_stream_failure_before_any_text_still_falls_back(gateway: GatewayFactory) -> None:
    """Failures before non-empty output keep the engine's provider fallback."""
    gw = gateway(adapters={"google": EmptyThenFailAdapter("google")})

    response = gw.create({"input": "hi", "stream": True})

    events = _sse_events(response.text)
    completed = events[-1][1]["response"]
    assert events[-1][0] == "response.completed"
    assert completed["model"] != "gemini-3.5-flash"
    assert completed["output"][0]["content"][0]["text"] == f"[{completed['model']}] routed response"
    assert "fallback attempt 1 succeeded" in completed["nexus"]["rationale"]


@pytest.mark.parametrize("adapter_type", [EmptyAnswerAdapter, SilentStreamAdapter])
def test_stream_with_empty_answer_completes_with_selected_model(
    gateway: GatewayFactory, adapter_type: type[MockProviderAdapter]
) -> None:
    """A successful stream without text (one empty chunk or none) reports its model."""
    gw = gateway(adapters={"google": adapter_type("google")})

    response = gw.create({"input": "hi", "stream": True})

    events = _sse_events(response.text)
    names = [name for name, _ in events]
    assert "response.output_text.delta" not in names
    assert names[-1] == "response.completed"
    assert events[0][1]["response"]["model"] == "gemini-3.5-flash"
    completed = events[-1][1]["response"]
    assert completed["model"] == "gemini-3.5-flash"
    assert completed["output"][0]["content"][0]["text"] == ""


def test_stream_reports_fallback_model_after_silent_fallback(gateway: GatewayFactory) -> None:
    """Lifecycle events name the fallback that answered, not the attempt that failed."""
    silent = {name: SilentStreamAdapter(name) for name in ("openai", "anthropic", "moonshot")}
    gw = gateway(adapters={"google": EmptyThenFailAdapter("google"), **silent})

    response = gw.create({"input": "hi", "stream": True})

    events = _sse_events(response.text)
    completed = events[-1][1]["response"]
    assert events[-1][0] == "response.completed"
    assert completed["model"] != "gemini-3.5-flash"
    assert events[0][1]["response"]["model"] == completed["model"]
    assert events[1][1]["response"]["model"] == completed["model"]
