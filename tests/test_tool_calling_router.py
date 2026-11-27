"""Tests for capability-aware routing, accounting, caching, and PII of tool-calling requests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.analyzer import estimate_prompt_tokens
from router.capability_catalog import ModelCapabilityCatalog
from router.config import RouterSettings
from router.engine import NexusRouter, ToolCallingUnsupportedError
from router.model_ids import GEMINI_PRO_MODEL
from router.schemas import (
    ChatMessage,
    ProviderResponse,
    RouterRequest,
    ToolCall,
    ToolCallFunction,
    ToolConfig,
)
from router.strategies import ToolCallingPreferStrategy
from safety.pii import PiiScrubber
from safety.request_fingerprint import RequestFingerprintDeduper

PROVIDERS = ("openai", "anthropic", "google", "moonshot")
CANARY_ARGUMENT = "canary-argument-1234"
CANARY_RESULT = "canary-result-5678"
TOOLS = ToolConfig.model_validate(
    {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "examples": [CANARY_ARGUMENT]}},
                        "required": ["city"],
                    },
                },
            }
        ]
    }
)


class RecordingAdapter(MockProviderAdapter):
    """Mock provider that records the ``tool_config`` each call received."""

    def __init__(self, provider_name: str) -> None:
        super().__init__(provider_name)
        self.tool_configs: list[ToolConfig | None] = []

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        tool_config: ToolConfig | None = None,
    ) -> ProviderResponse:
        self.tool_configs.append(tool_config)
        return await super().complete(model, messages, max_tokens, tool_config)


class LegacyAdapter(MockProviderAdapter):
    """Adapter written before tool calling: ``complete`` takes no ``tool_config``."""

    async def complete(  # type: ignore[override]
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        return await super().complete(model, messages, max_tokens)


def _router(
    tmp_path: Path,
    capability_catalog: ModelCapabilityCatalog | None = None,
    adapters: dict[str, MockProviderAdapter] | None = None,
) -> tuple[NexusRouter, dict[str, MockProviderAdapter]]:
    settings = RouterSettings(
        _env_file=None,
        response_cache_enabled=False,
        audit_log_path=str(tmp_path / "audit-log.jsonl"),
        spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
        idempotency_path=str(tmp_path / "idempotency.sqlite3"),
        feedback_path=str(tmp_path / "feedback.sqlite3"),
    )
    adapters = adapters or {name: RecordingAdapter(name) for name in PROVIDERS}
    registry = AdapterRegistry(dict(adapters))
    return NexusRouter(settings, registry, capability_catalog=capability_catalog), adapters


def _request(*messages: ChatMessage, tool_config: ToolConfig | None = TOOLS) -> RouterRequest:
    return RouterRequest(
        request_id=uuid4().hex,
        messages=list(messages) or [ChatMessage(content="What is the weather in Paris?")],
        tool_config=tool_config,
    )


def _replayed_turn() -> list[ChatMessage]:
    call = ToolCall(
        id="call_1",
        function=ToolCallFunction(name="get_weather", arguments=json.dumps({"city": "Paris"})),
    )
    return [
        ChatMessage(content="What is the weather in Paris?"),
        ChatMessage(role="assistant", content="", tool_calls=[call]),
        ChatMessage(role="tool", tool_call_id="call_1", content=CANARY_RESULT),
    ]


def _calls(adapters: dict[str, MockProviderAdapter]) -> int:
    return sum(
        len(adapter.tool_configs)
        for adapter in adapters.values()
        if isinstance(adapter, RecordingAdapter)
    )


async def test_tool_requests_skip_models_without_the_tools_capability(tmp_path: Path) -> None:
    """Only tool-capable candidates are tried, in strategy order; skipped ones are named."""
    catalog = ModelCapabilityCatalog(seed={GEMINI_PRO_MODEL: {"tools"}})
    router, adapters = _router(tmp_path, catalog)
    response = await router.complete(_request())
    assert response.model_used == GEMINI_PRO_MODEL
    assert "tool calling skipped" in response.rationale
    assert response.finish_reason == "tool_calls"
    assert [call.function.name for call in response.tool_calls] == ["get_weather"]
    assert _calls(adapters) == 1
    assert adapters["google"].tool_configs == [TOOLS]  # type: ignore[attr-defined]


@pytest.mark.parametrize("replayed", [False, True])
async def test_tool_requests_without_a_capable_model_are_refused_before_dispatch(
    tmp_path: Path, replayed: bool
) -> None:
    """An explicitly empty catalog means no model may call tools: nothing is dispatched."""
    router, adapters = _router(tmp_path, ModelCapabilityCatalog(auto_seed=False))
    request = _request(*_replayed_turn(), tool_config=None) if replayed else _request()
    with pytest.raises(ToolCallingUnsupportedError, match="supports tool calling"):
        await router.complete(request)
    with pytest.raises(ToolCallingUnsupportedError):
        async for _ in router.complete_stream(request):
            pass
    assert _calls(adapters) == 0


async def test_text_requests_ignore_tool_capabilities_and_keep_legacy_adapters_working(
    tmp_path: Path,
) -> None:
    """Text-only requests route as before, even through adapters without ``tool_config``."""
    adapters: dict[str, MockProviderAdapter] = {name: LegacyAdapter(name) for name in PROVIDERS}
    router, _ = _router(tmp_path, ModelCapabilityCatalog(auto_seed=False), adapters)
    text = _request(tool_config=None)
    response = await router.complete(text)
    assert response.content.endswith("routed response")
    streamed = [chunk async for _, chunk in router.complete_stream(text)]
    assert "".join(streamed).endswith("routed response")


async def test_streamed_tool_calls_reach_the_final_router_response(tmp_path: Path) -> None:
    """``complete_stream`` passes tools to the adapter and reports the calls on completion."""
    router, _ = _router(tmp_path)
    completed = []
    chunks = [
        chunk async for _, chunk in router.complete_stream(_request(), on_complete=completed.append)
    ]
    assert "".join(chunks) == ""
    assert completed[0].finish_reason == "tool_calls"
    assert [call.function.name for call in completed[0].tool_calls] == ["get_weather"]


async def test_audit_and_logs_keep_tool_names_and_counts_but_never_arguments_or_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Audit lines carry tool counts and called names; arguments and results never appear."""
    router, _ = _router(tmp_path)
    await router.complete(_request())
    await router.complete(_request(*_replayed_turn()))
    await router.complete(_request(tool_config=None))
    audit_text = (tmp_path / "audit-log.jsonl").read_text()
    records = [json.loads(line) for line in audit_text.splitlines()]
    assert [(r.get("tool_count"), r.get("tool_call_names")) for r in records] == [
        (1, ["get_weather"]),
        (1, []),
        (None, None),
    ]
    assert "tool_count" not in records[2]
    captured = capsys.readouterr()
    for text in (audit_text, captured.out, captured.err):
        assert CANARY_ARGUMENT not in text
        assert CANARY_RESULT not in text
    assert "get_weather" in captured.out + captured.err


def test_token_estimates_count_tool_definitions_and_replayed_calls() -> None:
    """Tool schemas and earlier calls are billed as prompt tokens, so they are estimated."""
    text_only = _request(tool_config=None)
    with_tools = _request()
    replayed = _request(*_replayed_turn(), tool_config=None)
    assert text_only.token_estimate_text == text_only.prompt_text
    assert estimate_prompt_tokens(with_tools.token_estimate_text) > estimate_prompt_tokens(
        with_tools.prompt_text
    )
    assert "get_weather" in replayed.token_estimate_text
    assert (text_only.uses_tools, with_tools.uses_tools, replayed.uses_tools) == (
        False,
        True,
        True,
    )


def test_tool_calling_prefer_strategy_treats_offered_tools_as_a_tool_request() -> None:
    """Real ``tools`` (or replayed calls) count as tool use, not only metadata hints."""
    assert ToolCallingPreferStrategy._wants_tools(_request())
    assert ToolCallingPreferStrategy._wants_tools(_request(*_replayed_turn(), tool_config=None))
    assert not ToolCallingPreferStrategy._wants_tools(_request(tool_config=None))


def _legacy_cache_key(messages: list[dict[str, str]], temperature: float) -> str:
    """The key earlier builds computed (no tool fields), for the given text messages."""
    payload = {
        "tenant": "",
        "model": "",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": None,
        "strategy": "",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_response_cache_keys_cover_tools_tool_choice_and_tool_messages() -> None:
    """Any tool input changes the key; text-only requests keep their earlier keys."""
    text = [ChatMessage(content="hi")]
    assert ResponseCache.make_key(model=None, messages=text, temperature=0.2) == (
        _legacy_cache_key([{"role": "user", "content": "hi"}], 0.2)
    )
    base = TOOLS.canonical()
    variants = [
        None,
        base,
        {**base, "tool_choice": "required"},
        {**base, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
        {**base, "parallel_tool_calls": False},
        {**base, "tools": [{**base["tools"][0], "function": {"name": "other"}}]},
    ]
    keys = {
        ResponseCache.make_key(model=None, messages=text, temperature=0.2, tools=variant)
        for variant in variants
    }
    assert len(keys) == len(variants)
    turn = _replayed_turn()
    other_result = [*turn[:2], turn[2].model_copy(update={"content": "18C"})]
    other_call_id = [
        turn[0],
        turn[1].model_copy(
            update={"tool_calls": [turn[1].tool_calls[0].model_copy(update={"id": "call_2"})]}  # type: ignore[index]
        ),
        turn[2].model_copy(update={"tool_call_id": "call_2"}),
    ]
    replay_keys = {
        ResponseCache.make_key(model=None, messages=messages, temperature=0.2)
        for messages in (turn, other_result, other_call_id, turn[:1])
    }
    assert len(replay_keys) == 4


def test_request_fingerprints_cover_tools_and_keep_text_digests() -> None:
    """The deduper hashes tool settings when given, without changing text-only digests."""
    messages = [{"role": "user", "content": "hi"}]
    legacy = hashlib.sha256(
        json.dumps(
            {"model": "m", "messages": messages, "temperature": None, "tenant": None},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    plain = RequestFingerprintDeduper.fingerprint(model="m", messages=messages)
    with_tools = RequestFingerprintDeduper.fingerprint(
        model="m", messages=messages, tools=TOOLS.canonical()
    )
    forced = RequestFingerprintDeduper.fingerprint(
        model="m", messages=messages, tools={**TOOLS.canonical(), "tool_choice": "required"}
    )
    assert plain == legacy
    assert len({plain, with_tools, forced}) == 3


def test_pii_scrubbing_covers_tool_arguments_and_results_and_keeps_json_valid() -> None:
    """Arguments and results are scrubbed leaf by leaf; ids and names are untouched."""
    scrubber = PiiScrubber(enabled=True)
    arguments = json.dumps({"to": "ada@example.com", "phone": 4155551234, "count": 3, "ok": True})
    call = ToolCall(id="call_1", function=ToolCallFunction(name="send_mail", arguments=arguments))
    messages = [
        ChatMessage(content="Mail ada@example.com"),
        ChatMessage(role="assistant", content="", tool_calls=[call]),
        ChatMessage(role="tool", tool_call_id="call_1", content='{"sent_to": ["ada@example.com"]}'),
        ChatMessage(role="tool", tool_call_id="call_1", content="Call (415) 555-1234 back"),
        ChatMessage(role="tool", tool_call_id="call_1", content='{"status":"ok"}'),
    ]
    scrubbed = scrubber.scrub_messages(messages)
    assert scrubbed[0].content == "Mail [REDACTED_EMAIL]"
    scrubbed_call = scrubbed[1].tool_calls[0]  # type: ignore[index]
    assert (scrubbed_call.id, scrubbed_call.function.name) == ("call_1", "send_mail")
    assert json.loads(scrubbed_call.function.arguments) == {
        "to": "[REDACTED_EMAIL]",
        "phone": "[REDACTED_PHONE]",
        "count": 3,
        "ok": True,
    }
    assert json.loads(scrubbed[2].content) == {"sent_to": ["[REDACTED_EMAIL]"]}
    assert scrubbed[3].content == "Call [REDACTED_PHONE] back"
    assert scrubbed[4].content == '{"status":"ok"}'  # no PII: forwarded byte for byte
    assert [message.tool_call_id for message in scrubbed[2:]] == ["call_1"] * 3
    assert PiiScrubber(enabled=False).scrub_messages(messages) is messages


def test_pii_scrubbing_keeps_every_entry_when_redacted_keys_collide() -> None:
    """Keys that redact to the same placeholder get distinct names; no entry is dropped."""
    scrubber = PiiScrubber(enabled=True)
    result = json.dumps(
        {
            "ada@example.com": {"balance": 10},
            "[REDACTED_EMAIL]_2": "existing",
            "bob@example.com": {"balance": 20},
            "[REDACTED_EMAIL]": "placeholder",
        }
    )
    [scrubbed] = scrubber.scrub_messages(
        [ChatMessage(role="tool", tool_call_id="call_1", content=result)]
    )
    assert json.loads(scrubbed.content) == {
        "[REDACTED_EMAIL]_3": {"balance": 10},
        "[REDACTED_EMAIL]_2": "existing",
        "[REDACTED_EMAIL]_4": {"balance": 20},
        "[REDACTED_EMAIL]": "placeholder",
    }
