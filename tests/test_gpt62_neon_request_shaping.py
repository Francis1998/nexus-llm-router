"""Provider request shaping for GPT-6.2 Astra / Sol and Gemini 4 Neon (2026-11-19).

OpenAI's GPT-6.2 models reject reasoning effort ``none`` (the migration guide
says to send ``low`` instead of ``none`` or ``minimal``) and require omitting
``temperature``, ``top_p``, ``logprobs`` and ``top_logprobs``; a request the
provider rejects with HTTP 400 stops the fallback chain, so one forwarded
sampling field would fail the whole request. Tool calling on GPT-6.2 needs the
Responses API, so the Chat Completions payload never carries tools. Gemini 4 Neon
gets the same Gemini 4 shaping as Carbon, Argon and Hydrogen: no sampling or
thinking overrides, only ``maxOutputTokens``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.main as main
from adapters.base import BaseProviderAdapter
from adapters.google import GoogleGeminiAdapter
from adapters.mock import MockProviderAdapter
from adapters.openai import OpenAIAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings
from router.engine import NexusRouter
from router.schemas import ChatMessage
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.spend_ledger import SpendLedger
from safety.tenant_rate_limiter import TenantRateLimiter

GPT_6_2_MODELS = ("gpt-6.2-astra", "gpt-6.2-sol")
GEMINI_4_MODELS = ("gemini-4-carbon", "gemini-4-argon", "gemini-4-neon", "gemini-4-hydrogen")
OPENAI_PAYLOAD_KEYS = {"model", "messages", "max_completion_tokens"}
NEVER_SENT_TO_GPT_6_2 = {
    "temperature",
    "top_p",
    "logprobs",
    "top_logprobs",
    "reasoning_effort",
    "reasoning",
    "service_tier",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "parallel_tool_calls",
}
# A general prompt that no rule-based rule claims, so an explicit model is honored.
GENERAL_PROMPT = (
    "Design a multi-region failover strategy for a payments platform: compare "
    "active-active, active-passive, and pilot-light architectures, analyze cost, "
    "latency, and consistency trade-offs, and recommend one with a phased rollout plan."
)

Payloads = list[dict[str, Any]]


def _response(body: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = body
    return response


@pytest.fixture
def openai_payloads() -> Iterator[Payloads]:
    """Capture every JSON body the OpenAI adapter posts (no network)."""
    payloads: Payloads = []

    async def fake_post(*args: object, **kwargs: object) -> MagicMock:
        del args
        payload = kwargs["json"]
        assert isinstance(payload, dict)
        payloads.append(payload)
        return _response(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }
        )

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = fake_post
    with patch("adapters.openai.httpx.AsyncClient", return_value=client):
        yield payloads


@pytest.fixture
def api_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, openai_payloads: Payloads
) -> Callable[[], TestClient]:
    """Serve the real API with the real OpenAI adapter and mock other providers."""

    def _build() -> TestClient:
        settings = RouterSettings(
            audit_log_path=str(tmp_path / "audit-log.jsonl"),
            spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
            idempotency_path=str(tmp_path / "idempotency.sqlite3"),
            response_cache_enabled=False,
        )
        adapters: dict[str, BaseProviderAdapter] = {
            name: MockProviderAdapter(name) for name in ("anthropic", "google", "moonshot")
        }
        adapters["openai"] = OpenAIAdapter(api_key="test-key", timeout_seconds=5.0)
        registry = AdapterRegistry(adapters)
        router = NexusRouter(settings, registry)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=60, refill_per_second=1.0),
        )
        ledger = SpendLedger(settings.spend_ledger_path)
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: ResponseCache(ttl_seconds=60.0))
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        monkeypatch.setattr(main, "get_spend_ledger", lambda: ledger)
        return TestClient(main.app)

    return _build


@pytest.mark.parametrize("model", GPT_6_2_MODELS)
async def test_openai_adapter_sends_only_model_messages_and_max_completion_tokens(
    model: str, openai_payloads: Payloads
) -> None:
    """No sampling, logprobs, reasoning-effort, service-tier or tool fields reach GPT-6.2."""
    adapter = OpenAIAdapter(api_key="test-key", timeout_seconds=5.0)

    await adapter.complete(
        model,
        [ChatMessage(role="system", content="Be terse."), ChatMessage(content="Hi.")],
        max_tokens=64,
    )

    (payload,) = openai_payloads
    assert set(payload) == OPENAI_PAYLOAD_KEYS
    assert payload["model"] == model
    assert payload["max_completion_tokens"] == 64


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "gpt-6.2-sol",
                "messages": [{"role": "user", "content": GENERAL_PROMPT}],
                "temperature": 0.2,
                "top_p": 0.5,
                "logprobs": True,
                "top_logprobs": 3,
                "reasoning_effort": "none",
                "service_tier": "flex",
            },
        ),
        (
            "/v1/chat/completions",
            {
                "model": "gpt-6.2-astra",
                "messages": [{"role": "user", "content": GENERAL_PROMPT}],
                "temperature": 0.0,
                "reasoning_effort": "minimal",
            },
        ),
        (
            "/v1/responses",
            {
                "model": "gpt-6.2-astra",
                "input": GENERAL_PROMPT,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_logprobs": 2,
                "reasoning": {"effort": "none"},
            },
        ),
        (
            "/v1/messages",
            {
                "model": "gpt-6.2-sol",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": GENERAL_PROMPT}],
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 5,
                "output_config": {"effort": "low"},
            },
        ),
    ],
    ids=["chat-effort-none", "chat-effort-minimal", "responses", "anthropic-messages"],
)
def test_client_sampling_and_effort_fields_never_reach_gpt_6_2(
    api_client: Callable[[], TestClient],
    openai_payloads: Payloads,
    path: str,
    body: dict[str, Any],
) -> None:
    """Every compatible API serves GPT-6.2 without forwarding fields it rejects."""
    response = api_client().post(path, json=body)

    assert response.status_code == 200, response.text
    assert response.json()["model"] == body["model"]
    (payload,) = openai_payloads
    assert payload["model"] == body["model"]
    assert set(payload) == OPENAI_PAYLOAD_KEYS
    assert NEVER_SENT_TO_GPT_6_2.isdisjoint(payload)


def test_responses_api_refuses_tools_instead_of_reaching_the_provider(
    api_client: Callable[[], TestClient], openai_payloads: Payloads
) -> None:
    """Tool calls are refused with a clear 400, never sent to GPT-6.2 as Chat Completions."""
    response = api_client().post(
        "/v1/responses",
        json={
            "model": "gpt-6.2-sol",
            "input": GENERAL_PROMPT,
            "tools": [{"type": "function", "name": "get_weather", "parameters": {}}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "tools"
    assert openai_payloads == []


@pytest.mark.parametrize("model", GEMINI_4_MODELS)
async def test_gemini_4_models_share_the_same_generate_content_shape(model: str) -> None:
    """Gemini 4 Neon gets the Gemini 4 payload: no sampling or thinking overrides."""
    captured: dict[str, Any] = {}

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return _response({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = fake_post
    adapter = GoogleGeminiAdapter(api_key="test-key", timeout_seconds=5.0)
    with patch("adapters.google.httpx.AsyncClient", return_value=client):
        await adapter.complete(
            model,
            [ChatMessage(role="system", content="Be terse."), ChatMessage(content="Hi.")],
            max_tokens=64,
        )

    assert captured["url"].endswith(f"/models/{model}:generateContent")
    payload = captured["json"]
    assert set(payload) == {"contents", "generationConfig", "systemInstruction"}
    assert payload["generationConfig"] == {"maxOutputTokens": 64}
