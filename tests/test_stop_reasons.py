"""Provider stop reasons propagate to Chat Completions and Responses (incomplete)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as main
from adapters.anthropic import AnthropicAdapter
from adapters.base import BaseProviderAdapter
from adapters.google import GoogleGeminiAdapter
from adapters.mock import MockProviderAdapter
from adapters.moonshot import MoonshotAdapter
from adapters.openai import OpenAIAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings
from router.engine import NexusRouter
from router.schemas import ChatMessage, ProviderResponse
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.tenant_rate_limiter import TenantRateLimiter

PROVIDERS = ("openai", "anthropic", "google", "moonshot")


class TruncatingAdapter(MockProviderAdapter):
    """Mock provider that reports its answer was cut at ``max_tokens``."""

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        response = await super().complete(model, messages, max_tokens)
        return response.model_copy(update={"output_tokens": max_tokens, "finish_reason": "length"})


@dataclass
class Gateway:
    """Test client wired to an isolated router."""

    client: TestClient

    def chat(self, body: dict[str, Any]) -> httpx.Response:
        return self.client.post("/v1/chat/completions", json=body)

    def responses(self, body: dict[str, Any]) -> httpx.Response:
        return self.client.post("/v1/responses", json=body)


GatewayFactory = Callable[..., Gateway]


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GatewayFactory:
    """Build an API client backed by the given adapters and tmp state files."""

    def _build(adapter_type: type[MockProviderAdapter] = MockProviderAdapter) -> Gateway:
        settings = RouterSettings(
            audit_log_path=str(tmp_path / "audit-log.jsonl"),
            spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
            idempotency_path=str(tmp_path / "idempotency.sqlite3"),
            response_cache_enabled=False,
        )
        registry = AdapterRegistry({name: adapter_type(name) for name in PROVIDERS})
        router = NexusRouter(settings, registry)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=60, refill_per_second=1.0),
        )
        cache = ResponseCache(ttl_seconds=60.0)
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: cache)
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        return Gateway(client=TestClient(main.app))

    return _build


def _sse_data(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _sse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.partition(": ")[::2] for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


ADAPTERS: dict[str, Callable[[], BaseProviderAdapter]] = {
    "openai": lambda: OpenAIAdapter(api_key="test-key", timeout_seconds=5.0),
    "moonshot": lambda: MoonshotAdapter(
        api_key="test-key", base_url="https://api.moonshot.ai/v1", timeout_seconds=5.0
    ),
    "anthropic": lambda: AnthropicAdapter(api_key="test-key", timeout_seconds=5.0),
    "google": lambda: GoogleGeminiAdapter(api_key="test-key", timeout_seconds=5.0),
}
# provider: (model, truncated body, natural-stop body)
PROVIDER_BODIES: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "openai": (
        "gpt-6-luna",
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    ),
    "moonshot": (
        "kimi-k3.1-pro",
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    ),
    "anthropic": (
        "claude-haiku-5-5",
        {"content": [{"type": "text", "text": "ok"}], "stop_reason": "max_tokens"},
        {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"},
    ),
    "google": (
        "gemini-3.8-flash",
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "MAX_TOKENS"}]},
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]},
    ),
}


async def _complete_with_body(provider: str, body: dict[str, Any]) -> ProviderResponse:
    model, _, _ = PROVIDER_BODIES[provider]
    adapter = ADAPTERS[provider]()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = body
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=response)
    with patch(f"adapters.{provider}.httpx.AsyncClient", return_value=client):
        return await adapter.complete(model, [ChatMessage(content="Hi.")], max_tokens=16)


@pytest.mark.parametrize("provider", sorted(PROVIDER_BODIES))
async def test_adapters_normalize_provider_stop_reasons(provider: str) -> None:
    """OpenAI/Kimi ``length``, Anthropic ``max_tokens``, Gemini ``MAX_TOKENS`` -> ``length``."""
    _, truncated_body, natural_body = PROVIDER_BODIES[provider]

    truncated = await _complete_with_body(provider, truncated_body)
    natural = await _complete_with_body(provider, natural_body)

    assert truncated.finish_reason == "length"
    assert natural.finish_reason == "stop"


def test_chat_completion_reports_length_when_max_tokens_truncates(gateway: GatewayFactory) -> None:
    """The mock answers 16 tokens, so ``max_tokens=8`` truncates and ``64`` does not."""
    client = gateway()
    messages = [{"role": "user", "content": "Summarize the release notes."}]

    truncated = client.chat({"messages": messages, "max_tokens": 8}).json()
    natural = client.chat({"messages": messages, "max_tokens": 64}).json()

    assert truncated["choices"][0]["finish_reason"] == "length"
    assert natural["choices"][0]["finish_reason"] == "stop"


def test_chat_stream_final_chunk_reports_length(gateway: GatewayFactory) -> None:
    """The last SSE chunk carries the provider stop reason instead of a fixed ``stop``."""
    client = gateway()
    body = {
        "messages": [{"role": "user", "content": "Summarize the release notes."}],
        "max_tokens": 8,
        "stream": True,
    }

    chunks = _sse_data(client.chat(body).text)

    assert [chunk["choices"][0]["finish_reason"] for chunk in chunks][-1] == "length"
    assert all(chunk["choices"][0]["finish_reason"] is None for chunk in chunks[:-1])


def test_responses_truncation_is_incomplete(gateway: GatewayFactory) -> None:
    """A provider cut at ``max_output_tokens`` yields ``status: incomplete``."""
    client = gateway(TruncatingAdapter)

    response = client.responses({"input": "Summarize the release notes.", "max_output_tokens": 16})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    assert body["completed_at"] is None
    assert body["output"][0]["status"] == "incomplete"
    assert body["usage"]["output_tokens"] == 16


def test_responses_complete_answers_stay_completed(gateway: GatewayFactory) -> None:
    """Answers that end naturally keep ``status: completed`` and no incomplete details."""
    client = gateway()

    body = client.responses({"input": "Summarize the release notes."}).json()

    assert body["status"] == "completed"
    assert body["incomplete_details"] is None
    assert body["output"][0]["status"] == "completed"


def test_responses_stream_ends_with_response_incomplete(gateway: GatewayFactory) -> None:
    """Streams cut at ``max_output_tokens`` end with ``response.incomplete``."""
    client = gateway(TruncatingAdapter)
    body = {"input": "Summarize the release notes.", "max_output_tokens": 16, "stream": True}

    events = _sse_events(client.responses(body).text)
    types = [event_type for event_type, _ in events]

    assert types[-1] == "response.incomplete"
    assert "response.completed" not in types
    assert [data["sequence_number"] for _, data in events] == list(range(len(events)))
    item_done = next(
        data for event_type, data in events if event_type == "response.output_item.done"
    )
    assert item_done["item"]["status"] == "incomplete"
    final = events[-1][1]["response"]
    assert final["status"] == "incomplete"
    assert final["incomplete_details"] == {"reason": "max_output_tokens"}


def test_responses_stream_reports_provider_usage(gateway: GatewayFactory) -> None:
    """Streaming usage is the provider-reported usage, identical to the non-streaming call."""
    client = gateway()
    body = {"input": "Summarize the release notes.", "max_output_tokens": 64}

    plain = client.responses(body).json()
    events = _sse_events(client.responses(body | {"stream": True}).text)

    assert events[-1][0] == "response.completed"
    assert events[-1][1]["response"]["usage"] == plain["usage"]
