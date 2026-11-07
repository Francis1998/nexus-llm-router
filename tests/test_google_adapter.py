"""Tests for Google Gemini adapter system-message handling."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as main
from adapters.google import GoogleGeminiAdapter, _extract_gemini_text
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings
from router.engine import NexusRouter
from router.model_ids import GEMINI_FLASH_MODEL
from router.schemas import ChatMessage
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.tenant_rate_limiter import TenantRateLimiter

GEMINI_KEY_CANARY = "nexus-gemini-key-leak-canary-0123456789"


def test_extract_gemini_text_skips_thought_summary_parts() -> None:
    """Thought-summary parts (``thought: true``) must not leak into the answer.

    Gemini 2.5/3-series thinking models interleave thought parts \u2014 marked
    ``{"thought": true}`` and carrying internal reasoning \u2014 with the answer
    parts. Concatenating every ``text`` part surfaced that reasoning in the
    user-facing completion; only non-thought answer text must be returned.
    """
    candidates = [
        {
            "content": {
                "parts": [
                    {"thought": True, "text": "The user wants a greeting; I will be brief."},
                    {"text": "Hello, "},
                    {"text": "world."},
                ]
            }
        }
    ]

    assert _extract_gemini_text(candidates) == "Hello, world."


@pytest.mark.asyncio()
async def test_complete_posts_system_instruction_to_gemini() -> None:
    """complete() must map system messages to Gemini's systemInstruction field."""
    adapter = GoogleGeminiAdapter(api_key="test-key", timeout_seconds=5.0)
    captured: dict[str, Any] = {}

    async def fake_post(*args: object, **kwargs: object) -> MagicMock:
        del args
        captured.update(kwargs)
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }
        return response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("adapters.google.httpx.AsyncClient", return_value=mock_client):
        await adapter.complete(
            GEMINI_FLASH_MODEL,
            [
                ChatMessage(role="system", content="Safety policy."),
                ChatMessage(role="user", content="Hello."),
            ],
            max_tokens=32,
        )

    payload = captured["json"]
    assert payload["systemInstruction"] == {"parts": [{"text": "Safety policy."}]}
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "Hello."}]}]


@pytest.mark.asyncio()
async def test_complete_concatenates_all_gemini_text_parts() -> None:
    """complete() must join every text part, not only the first.

    Gemini's ``candidates[0].content.parts`` is a list that can hold multiple
    text segments (and non-text parts). Reading only ``parts[0]`` silently
    truncated multi-part completions. Non-text parts must be skipped and all
    text parts concatenated in order.
    """
    adapter = GoogleGeminiAdapter(api_key="test-key", timeout_seconds=5.0)

    async def fake_post(*args: object, **kwargs: object) -> MagicMock:
        del args, kwargs
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Hello, "},
                            {"functionCall": {"name": "noop"}},
                            {"text": "world."},
                        ]
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
        }
        return response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("adapters.google.httpx.AsyncClient", return_value=mock_client):
        result = await adapter.complete(
            GEMINI_FLASH_MODEL,
            [ChatMessage(role="user", content="Hi.")],
            max_tokens=32,
        )

    assert result.content == "Hello, world."


def _gemini_client_factory(requests: list[httpx.Request]) -> Callable[..., httpx.AsyncClient]:
    """Return a factory for real ``httpx.AsyncClient``s that answer Gemini calls in-process.

    The clients are real, so httpx's INFO request log runs exactly as in production.
    """
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )

    def factory(*, timeout: float) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(handler), timeout=timeout)

    return factory


@pytest.mark.asyncio()
async def test_complete_sends_api_key_in_header_not_in_url() -> None:
    """The API key must travel in the ``x-goog-api-key`` header, never in the URL.

    A ``?key=`` query parameter copies the secret into everything that records
    request URLs (httpx's INFO request log, proxies, tracing). Google's guidance
    is to send API keys in the ``x-goog-api-key`` header.
    """
    adapter = GoogleGeminiAdapter(api_key=GEMINI_KEY_CANARY, timeout_seconds=5.0)
    requests: list[httpx.Request] = []

    with patch("adapters.google.httpx.AsyncClient", _gemini_client_factory(requests)):
        result = await adapter.complete(
            GEMINI_FLASH_MODEL, [ChatMessage(role="user", content="Hi.")], max_tokens=32
        )

    assert result.content == "ok"
    (request,) = requests
    assert request.headers.get("x-goog-api-key") == GEMINI_KEY_CANARY
    assert "key" not in request.url.params
    assert GEMINI_KEY_CANARY not in str(request.url)


def test_gateway_logs_never_contain_the_gemini_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A chat completion routed to Gemini must not write the provider key to the logs.

    httpx logs every request URL at INFO and the gateway's ``configure_logging``
    sends INFO records to stdout, so a key in the URL leaked on every Gemini call.
    """
    settings = RouterSettings(
        audit_log_path=str(tmp_path / "audit-log.jsonl"),
        spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
        idempotency_path=str(tmp_path / "idempotency.sqlite3"),
        failover_priority=[GEMINI_FLASH_MODEL],
    )
    registry = AdapterRegistry(
        {"google": GoogleGeminiAdapter(api_key=GEMINI_KEY_CANARY, timeout_seconds=5.0)}
    )
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
    caplog.set_level(logging.INFO, logger="httpx")
    requests: list[httpx.Request] = []

    with patch("adapters.google.httpx.AsyncClient", _gemini_client_factory(requests)):
        response = TestClient(main.app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi."}]},
            headers={"X-Router-Strategy": "failover-priority"},
        )

    assert response.status_code == 200
    assert response.json()["model"] == GEMINI_FLASH_MODEL
    assert len(requests) == 1
    provider_log_lines = [
        record.getMessage()
        for record in caplog.records
        if "generativelanguage.googleapis.com" in record.getMessage()
    ]
    assert provider_log_lines, "httpx should log the Gemini request"
    assert GEMINI_KEY_CANARY not in caplog.text
