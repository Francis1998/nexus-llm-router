"""Tests for composed gateway guards (idempotency + tenant rate limit)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.main import (
    app,
    get_gateway_guards,
    get_idempotency_store,
    get_response_cache,
    get_router,
    get_settings,
    get_tenant_rate_limiter,
)
from router.config import load_settings
from router.schemas import RouterResponse, RoutingStrategyName
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.tenant_rate_limiter import TenantRateLimiter, TenantRateLimitExceededError


def _sample_router_response(*, content: str = "guarded-answer") -> RouterResponse:
    return RouterResponse(
        content=content,
        model_used="gpt-5.5",
        routing_strategy=RoutingStrategyName.RULE_BASED,
        latency_ms=12.0,
        input_tokens=4,
        output_tokens=8,
        cost_usd=0.001,
        rationale="unit-test",
        request_id="req-guard-1",
    )


def test_gateway_guard_service_composes_store_and_limiter(tmp_path: Path) -> None:
    """Service delegates check/remember/assert to the composed building blocks."""
    store = IdempotencyStore(tmp_path / "idem.sqlite3", ttl_seconds=60.0)
    limiter = TenantRateLimiter(capacity=2, refill_per_second=0.001)
    guards = GatewayGuardService(store, limiter)

    assert guards.idempotency_store is store
    assert guards.tenant_rate_limiter is limiter
    assert guards.check_idempotency("k1", "acme") is None
    body = '{"id":"chatcmpl-1","model":"gpt-5.5"}'
    assert guards.remember_response("k1", "acme", body, 200) is True
    record = guards.check_idempotency("k1", "acme")
    assert record is not None
    assert record.response_json == body
    assert guards.remember_response("k1", "acme", '{"other":true}', 200) is False

    guards.assert_tenant_allowed("acme")
    guards.assert_tenant_allowed("acme")
    with pytest.raises(TenantRateLimitExceededError):
        guards.assert_tenant_allowed("acme")


def test_gateway_guard_service_scopes_idempotency_by_tenant() -> None:
    """Idempotency keys remain tenant-scoped through the facade."""
    store = IdempotencyStore(":memory:")
    limiter = TenantRateLimiter(capacity=10, refill_per_second=1.0)
    guards = GatewayGuardService(store, limiter)
    assert guards.remember_response("shared", "acme", '{"t":"acme"}', 200) is True
    assert guards.remember_response("shared", "globex", '{"t":"globex"}', 200) is True
    acme = guards.check_idempotency("shared", "acme")
    globex = guards.check_idempotency("shared", "globex")
    assert acme is not None and globex is not None
    assert json.loads(acme.response_json)["t"] == "acme"
    assert json.loads(globex.response_json)["t"] == "globex"


def _restore_api_getters() -> None:
    """Restore FastAPI module getters after other API tests rebound them."""
    import api.main as main

    main.get_settings = get_settings
    main.get_router = get_router
    main.get_response_cache = get_response_cache
    main.get_idempotency_store = get_idempotency_store
    main.get_tenant_rate_limiter = get_tenant_rate_limiter
    main.get_gateway_guards = get_gateway_guards
    load_settings.cache_clear()
    get_settings.cache_clear()
    get_router.cache_clear()
    get_response_cache.cache_clear()
    get_idempotency_store.cache_clear()
    get_tenant_rate_limiter.cache_clear()
    get_gateway_guards.cache_clear()


def test_api_getters_wire_gateway_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """lru_cache getters compose store + limiter from RouterSettings."""
    _restore_api_getters()
    monkeypatch.setenv("NEXUS_IDEMPOTENCY_PATH", str(tmp_path / "api-idem.sqlite3"))
    monkeypatch.setenv("NEXUS_IDEMPOTENCY_TTL_SECONDS", "120")
    monkeypatch.setenv("NEXUS_TENANT_RATE_LIMIT_CAPACITY", "5")
    monkeypatch.setenv("NEXUS_TENANT_RATE_LIMIT_REFILL_PER_SECOND", "2.5")
    load_settings.cache_clear()
    get_settings.cache_clear()
    get_idempotency_store.cache_clear()
    get_tenant_rate_limiter.cache_clear()
    get_gateway_guards.cache_clear()

    store = get_idempotency_store()
    limiter = get_tenant_rate_limiter()
    guards = get_gateway_guards()
    assert store.ttl_seconds == 120.0
    assert limiter.capacity == 5.0
    assert limiter.refill_per_second == 2.5
    assert guards.idempotency_store is store
    assert guards.tenant_rate_limiter is limiter


def test_chat_completions_replays_idempotency_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second request with the same Idempotency-Key skips provider dispatch."""
    _restore_api_getters()
    monkeypatch.setenv("NEXUS_IDEMPOTENCY_PATH", str(tmp_path / "replay-idem.sqlite3"))
    load_settings.cache_clear()
    get_settings.cache_clear()
    get_idempotency_store.cache_clear()
    get_gateway_guards.cache_clear()
    get_response_cache.cache_clear()
    cache = get_response_cache()
    cache.clear()
    mock_complete = AsyncMock(return_value=_sample_router_response(content="first-dispatch"))
    key = f"retry-{uuid4().hex}"
    with patch.object(get_router(), "complete", mock_complete):
        client = TestClient(app)
        body = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": f"idempotency please {key}"}],
            "temperature": 0.0,
            "user": "tenant-idem",
        }
        headers = {"Idempotency-Key": key}
        first = client.post("/v1/chat/completions", json=body, headers=headers)
        second = client.post("/v1/chat/completions", json=body, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["choices"][0]["message"]["content"] == "first-dispatch"
    assert second.json()["choices"][0]["message"]["content"] == "first-dispatch"
    assert mock_complete.await_count == 1


def test_chat_completions_tenant_rate_limit_returns_429() -> None:
    """X-Tenant-Id traffic is hard-limited; absent header stays unlimited."""
    _restore_api_getters()
    import api.main as main

    store = IdempotencyStore(":memory:")
    limiter = TenantRateLimiter(capacity=1, refill_per_second=0.0001)
    guards = GatewayGuardService(store, limiter)
    main.get_gateway_guards = lambda: guards  # type: ignore[assignment]
    cache = get_response_cache()
    cache.clear()
    mock_complete = AsyncMock(return_value=_sample_router_response())
    with patch.object(get_router(), "complete", mock_complete):
        client = TestClient(app)
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "rate me"}],
            "stream": False,
        }
        # Distinct body so the no-header request cannot hit the response cache.
        bypass_body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "rate me without tenant header"}],
            "stream": False,
        }
        ok = client.post("/v1/chat/completions", json=body, headers={"X-Tenant-Id": "noisy"})
        limited = client.post(
            "/v1/chat/completions", json=body, headers={"X-Tenant-Id": "noisy"}
        )
        bypass = client.post("/v1/chat/completions", json=bypass_body)
    assert ok.status_code == 200
    assert limited.status_code == 429
    assert bypass.status_code == 200
    assert mock_complete.await_count == 2
