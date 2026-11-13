"""Regression tests: budget refusals return the documented 402 on every API surface.

The budget guardrail refuses each candidate before dispatch, and the router used
to report a request whose every candidate it refused as "all provider attempts
failed" (HTTP 503 ``routing_failed`` / Anthropic ``overloaded_error``) instead of
the documented 402 ``budget_exceeded`` / ``billing_error``. Clients then retried a
request that could never succeed until the budget allowed it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as main
from adapters.base import BaseProviderAdapter
from adapters.mock import MockProviderAdapter
from adapters.openai import OpenAIAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings, default_model_catalog
from router.engine import NexusRouter
from router.model_ids import OPENAI_BALANCED_MODEL, OPENAI_FRONTIER_MODEL
from router.schemas import DomainTag, ModelCandidate
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.spend_ledger import SpendLedger
from safety.tenant_rate_limiter import TenantRateLimiter

PROMPT = "Summarize this infrastructure incident for the on-call channel."
BUDGET_REFUSAL = "budget cap exceeded for alice"
# A catalog row with placeholder $0 prices that no adapter price table lists.
UNPRICED = "ft:gpt-6.1-terra:acme:triage:9d1x0b7e"

# (path, request body) per API surface; every body bills the end user ``alice``.
SURFACES: dict[str, tuple[str, dict[str, Any]]] = {
    "chat": (
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": PROMPT}], "user": "alice"},
    ),
    "responses": ("/v1/responses", {"input": PROMPT, "user": "alice"}),
    "messages": (
        "/v1/messages",
        {
            "model": "claude-opus-5-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": PROMPT}],
            "metadata": {"user_id": "alice"},
        },
    ),
}


def _error_fields(surface: str, response: httpx.Response) -> tuple[str | None, str, str | None]:
    """Return ``(type, message, code)`` from the surface's error envelope."""
    body = response.json()
    if surface == "chat":
        return None, body["detail"], None
    if surface == "messages":
        assert body["type"] == "error"
        return body["error"]["type"], body["error"]["message"], None
    return body["error"]["type"], body["error"]["message"], body["error"]["code"]


def _sse_frames(body: str) -> list[dict[str, Any] | str]:
    frames: list[dict[str, Any] | str] = []
    for block in body.split("\n\n"):
        line = block.strip()
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            frames.append(payload if payload == "[DONE]" else json.loads(payload))
    return frames


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., TestClient]:
    """Build an API client over mock providers with the given router settings."""

    def build(
        *,
        adapters: dict[str, BaseProviderAdapter] | None = None,
        extra_models: list[ModelCandidate] | None = None,
        **overrides: object,
    ) -> TestClient:
        settings = RouterSettings(
            audit_log_path=str(tmp_path / "audit-log.jsonl"),
            spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
            idempotency_path=str(tmp_path / "idempotency.sqlite3"),
            **overrides,
        )
        registry = AdapterRegistry(
            {
                name: MockProviderAdapter(name)
                for name in ("openai", "anthropic", "google", "moonshot")
            }
            | (adapters or {})
        )
        catalog = default_model_catalog() | {
            candidate.model: candidate for candidate in extra_models or []
        }
        router = NexusRouter(settings, registry, model_catalog=catalog)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=60, refill_per_second=1.0),
        )
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: ResponseCache(enabled=False))
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        monkeypatch.setattr(
            main, "get_spend_ledger", lambda: SpendLedger(settings.spend_ledger_path)
        )
        return TestClient(main.app)

    return build


@pytest.mark.parametrize(
    ("surface", "stream", "error_type", "code"),
    [
        pytest.param("chat", False, None, None, id="chat"),
        pytest.param("responses", False, "budget_exceeded", "budget_exceeded", id="responses"),
        pytest.param(
            "responses", True, "budget_exceeded", "budget_exceeded", id="responses-stream"
        ),
        pytest.param("messages", False, "billing_error", None, id="messages"),
        pytest.param("messages", True, "billing_error", None, id="messages-stream"),
    ],
)
def test_budget_refusal_of_every_candidate_returns_402(
    gateway: Callable[..., TestClient],
    surface: str,
    stream: bool,
    error_type: str | None,
    code: str | None,
) -> None:
    """A user over budget gets the documented 402, not a 503 routing failure."""
    client = gateway(budget_cap_usd=0.0)
    path, body = SURFACES[surface]

    response = client.post(path, json=body | ({"stream": True} if stream else {}))

    assert response.status_code == 402
    assert _error_fields(surface, response) == (error_type, BUDGET_REFUSAL, code)


def test_budget_refusal_on_a_chat_stream_sends_a_budget_exceeded_frame(
    gateway: Callable[..., TestClient],
) -> None:
    """Chat streams report the refusal as a ``budget_exceeded`` error frame."""
    client = gateway(budget_cap_usd=0.0)
    path, body = SURFACES["chat"]

    response = client.post(path, json=body | {"stream": True})

    assert response.status_code == 200
    assert _sse_frames(response.text) == [
        {"error": {"message": BUDGET_REFUSAL, "type": "budget_exceeded"}},
        "[DONE]",
    ]


def test_provider_failure_after_a_budget_refusal_stays_a_routing_failure(
    gateway: Callable[..., TestClient],
) -> None:
    """Only a refusal of every candidate is a 402; a failed affordable one is a 503.

    GPT-6 Astra is over the $0.001 cap, GPT-6 Luna fits but its provider fails, so
    the client should retry later rather than raise its budget.
    """
    client = gateway(
        budget_cap_usd=0.001,
        failover_priority=[OPENAI_FRONTIER_MODEL, OPENAI_BALANCED_MODEL],
        adapters={"openai": MockProviderAdapter("openai", fail=True)},
    )
    path, body = SURFACES["responses"]

    response = client.post(path, json=body, headers={"X-Router-Strategy": "failover-priority"})

    assert response.status_code == 503
    error_type, message, code = _error_fields("responses", response)
    assert (error_type, code) == ("routing_failed", "routing_failed")
    assert "mock provider failed: openai" in message


@pytest.mark.parametrize(
    ("surface", "error_type", "code"),
    [
        pytest.param("chat", None, None, id="chat"),
        pytest.param("responses", "server_error", "model_price_unknown", id="responses"),
        pytest.param("messages", "api_error", None, id="messages"),
    ],
)
def test_request_for_models_without_any_price_returns_500_model_price_unknown(
    gateway: Callable[..., TestClient],
    surface: str,
    error_type: str | None,
    code: str | None,
) -> None:
    """No candidate can be priced: a gateway configuration error, not a budget overrun."""
    unpriced = ModelCandidate(
        model=UNPRICED,
        provider="openai",
        quality_score=0.9,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        supports_domains={DomainTag.GENERAL},
    )
    client = gateway(
        failover_priority=[UNPRICED],
        extra_models=[unpriced],
        adapters={"openai": OpenAIAdapter("test-key", 5.0)},
    )
    path, body = SURFACES[surface]

    response = client.post(path, json=body, headers={"X-Router-Strategy": "failover-priority"})

    assert response.status_code == 500
    actual_type, message, actual_code = _error_fields(surface, response)
    assert (actual_type, actual_code) == (error_type, code)
    assert f"model {UNPRICED} has no price" in message
