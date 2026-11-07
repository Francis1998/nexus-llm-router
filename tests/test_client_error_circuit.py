"""Regression tests: client-caused provider rejections must not open circuit breakers.

Before the fix, a provider's HTTP 400 for a request it could not accept (an
empty ``messages`` array, a system-only conversation, an over-long prompt)
counted as a provider failure and triggered cross-provider fallback. Three such
requests from any client opened the circuit breaker of every provider in the
fallback chain, so the gateway answered 503 to every tenant for 60 seconds.

The real provider adapters run against an in-process ``httpx.MockTransport``
that answers like the provider APIs do: HTTP 400 for a conversation without
user/assistant turns or for a prompt marked as unacceptable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as main
from adapters.anthropic import AnthropicAdapter
from adapters.base import BaseProviderAdapter, ProviderError
from adapters.google import GoogleGeminiAdapter
from adapters.moonshot import MoonshotAdapter
from adapters.openai import OpenAIAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings
from router.engine import NexusRouter
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_FRONTIER_MODEL
from router.schemas import ChatMessage
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.tenant_rate_limiter import TenantRateLimiter

PROVIDERS = ("openai", "anthropic", "google", "moonshot")
PROVIDER_BY_HOST = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "google",
    "api.moonshot.ai": "moonshot",
}
OPENAI_COMPATIBLE_OK = {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}
SUCCESS_BODIES: dict[str, dict[str, Any]] = {
    "openai": OPENAI_COMPATIBLE_OK,
    "moonshot": OPENAI_COMPATIBLE_OK,
    "anthropic": {
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 1},
    },
    "google": {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
    },
}
# Stands in for a prompt every provider refuses (for example one longer than the
# context window); the providers answer it with HTTP 400.
REJECTED_PROMPT = "PROMPT-THE-PROVIDER-REJECTS"
VALID_CHAT = {"messages": [{"role": "user", "content": "hi"}]}
REJECTED_CHAT = {"messages": [{"role": "user", "content": REJECTED_PROMPT}]}
FAILOVER = {"X-Router-Strategy": "failover-priority"}


@dataclass
class ProviderStub:
    """Answers provider HTTP calls in-process and records which providers were called."""

    calls: list[str] = field(default_factory=list)
    forced_status: dict[str, int] = field(default_factory=dict)

    def handle(self, request: httpx.Request) -> httpx.Response:
        provider = PROVIDER_BY_HOST[request.url.host]
        self.calls.append(provider)
        if provider in self.forced_status:
            return httpx.Response(self.forced_status[provider], json={"error": {"message": "x"}})
        payload = json.loads(request.content)
        turns = payload.get("contents", payload.get("messages"))
        if not turns or REJECTED_PROMPT in request.content.decode():
            return httpx.Response(400, json={"error": {"message": "invalid request"}})
        return httpx.Response(200, json=SUCCESS_BODIES[provider])

    def client_factory(self) -> Callable[..., httpx.AsyncClient]:
        real_async_client = httpx.AsyncClient

        def factory(*, timeout: float) -> httpx.AsyncClient:
            return real_async_client(transport=httpx.MockTransport(self.handle), timeout=timeout)

        return factory


def _real_adapters() -> dict[str, BaseProviderAdapter]:
    return {
        "openai": OpenAIAdapter("test-key", 5.0),
        "anthropic": AnthropicAdapter("test-key", 5.0),
        "google": GoogleGeminiAdapter("test-key", 5.0),
        "moonshot": MoonshotAdapter("test-key", "https://api.moonshot.ai/v1", 5.0),
    }


@dataclass
class Gateway:
    """API client wired to real adapters that talk to a provider stub."""

    client: TestClient
    router: NexusRouter
    stub: ProviderStub

    def chat(self, body: dict[str, Any], **headers: str) -> httpx.Response:
        return self.client.post("/v1/chat/completions", json=body, headers=headers)

    def respond(self, body: dict[str, Any], **headers: str) -> httpx.Response:
        return self.client.post("/v1/responses", json=body, headers=headers)

    def all_providers_routable(self) -> bool:
        return all(self.router.provider_routable(provider) for provider in PROVIDERS)


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Gateway:
    """Build the API with real provider adapters and an in-process provider stub."""
    stub = ProviderStub()
    monkeypatch.setattr(httpx, "AsyncClient", stub.client_factory())
    settings = RouterSettings(
        audit_log_path=str(tmp_path / "audit-log.jsonl"),
        spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
        idempotency_path=str(tmp_path / "idempotency.sqlite3"),
    )
    registry = AdapterRegistry(_real_adapters())
    router = NexusRouter(settings, registry)
    guards = GatewayGuardService(
        idempotency_store=IdempotencyStore(settings.idempotency_path),
        tenant_rate_limiter=TenantRateLimiter(capacity=60, refill_per_second=1.0),
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
    monkeypatch.setattr(main, "get_router", lambda: router)
    monkeypatch.setattr(main, "get_response_cache", lambda: ResponseCache(enabled=False))
    monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
    return Gateway(client=TestClient(main.app), router=router, stub=stub)


def _openai_error(response: httpx.Response, status: int) -> dict[str, Any]:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["type"] == "invalid_request_error"
    return dict(error)


def test_three_empty_message_requests_cannot_take_the_gateway_down(gateway: Gateway) -> None:
    """Malformed requests from one client must not 503 everyone else's traffic."""
    rejected = [gateway.chat({"messages": []}) for _ in range(3)]

    valid = gateway.chat(VALID_CHAT)

    assert valid.status_code == 200, valid.text
    assert [response.status_code for response in rejected] == [400, 400, 400]
    assert gateway.all_providers_routable()


@pytest.mark.parametrize(
    ("messages", "code"),
    [
        ([], "empty_array"),
        ([{"role": "system", "content": "Answer tersely."}], "invalid_value"),
    ],
)
def test_chat_rejects_conversations_no_provider_can_serve(
    gateway: Gateway, messages: list[dict[str, str]], code: str
) -> None:
    """Empty and system-only conversations get an OpenAI-style 400 before routing."""
    error = _openai_error(gateway.chat({"messages": messages}), 400)

    assert error["param"] == "messages"
    assert error["code"] == code
    assert gateway.stub.calls == []


def test_responses_rejects_system_only_input(gateway: Gateway) -> None:
    """``/v1/responses`` input with only developer/system items is a 400 before routing."""
    body = {
        "instructions": "Answer tersely.",
        "input": [{"role": "developer", "content": "Never apologize."}],
    }

    error = _openai_error(gateway.respond(body), 400)

    assert error["param"] == "input"
    assert error["code"] == "invalid_value"
    assert gateway.stub.calls == []


@pytest.mark.parametrize("status", [400, 413, 422])
def test_provider_rejection_is_a_4xx_without_fallback_or_circuit_failure(
    gateway: Gateway, status: int
) -> None:
    """A provider's rejection of the request itself goes back to the caller as-is.

    It must not be retried on other providers or recorded as a provider fault, so
    repeating it cannot open the circuit of the provider that answered it.
    """
    gateway.stub.forced_status["openai"] = status

    responses = [gateway.chat(VALID_CHAT, **FAILOVER) for _ in range(3)]

    assert [response.status_code for response in responses] == [status] * 3
    assert "rejected" in responses[0].json()["detail"]
    assert gateway.stub.calls == ["openai", "openai", "openai"]
    assert gateway.all_providers_routable()
    gateway.stub.forced_status.clear()
    assert gateway.chat(VALID_CHAT, **FAILOVER).json()["model"] == OPENAI_FRONTIER_MODEL


def test_responses_returns_provider_rejection_as_openai_400(gateway: Gateway) -> None:
    """``/v1/responses`` reports a provider rejection as an ``invalid_request_error``."""
    response = gateway.respond({"input": REJECTED_PROMPT}, **FAILOVER)

    error = _openai_error(response, 400)
    assert error["code"] == "provider_rejected_request"
    assert gateway.stub.calls == ["openai"]
    assert gateway.all_providers_routable()


def test_streamed_provider_rejection_is_not_a_provider_failure(gateway: Gateway) -> None:
    """Streaming chat reports the rejection in-band without fallback or circuit damage."""
    response = gateway.chat({**REJECTED_CHAT, "stream": True}, **FAILOVER)

    first_event = json.loads(response.text.split("\n\n")[0].removeprefix("data: "))
    assert first_event["error"]["type"] == "invalid_request_error"
    assert gateway.stub.calls == ["openai"]
    for _ in range(2):
        gateway.chat({**REJECTED_CHAT, "stream": True}, **FAILOVER)
    assert gateway.all_providers_routable()


@pytest.mark.parametrize("status", [500, 503, 408, 409, 429, 401, 403, 404])
def test_provider_faults_still_count_and_fall_back(gateway: Gateway, status: int) -> None:
    """5xx, timeouts, conflicts, rate limits and credential/model errors keep failing over."""
    gateway.stub.forced_status["openai"] = status

    response = gateway.chat(VALID_CHAT, **FAILOVER)

    assert response.status_code == 200, response.text
    assert response.json()["model"] == ANTHROPIC_SAFETY_MODEL
    assert gateway.stub.calls == ["openai", "anthropic"]
    assert gateway.router._circuit_breakers._states["openai"].consecutive_failures == 1


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_adapters_attach_the_http_status_to_provider_errors(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """Every adapter exposes the provider's HTTP status on ``ProviderError.status_code``."""
    stub = ProviderStub(forced_status={provider: 413})
    monkeypatch.setattr(httpx, "AsyncClient", stub.client_factory())
    adapter = _real_adapters()[provider]

    with pytest.raises(ProviderError) as raised:
        await adapter.complete("model-id", [ChatMessage(content="hi")], max_tokens=16)

    assert getattr(raised.value, "status_code", None) == 413
