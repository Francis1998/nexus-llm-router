"""Regression tests: the raw ``Authorization`` header must never become an identifier.

Before the fix, ``Bearer sk-...`` was used verbatim as ``RouterRequest.api_key_id``,
the ``user_id`` fallback, the spend-ledger tenant, and the idempotency / cache
tenant, so the secret leaked into HTTP 429/503 error details, ``GET /v1/spend``
and the durable SQLite state files.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as main
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings
from router.engine import NexusRouter
from router.schemas import RouterRequest, RouterResponse
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.spend_ledger import SpendLedger
from safety.tenant_rate_limiter import TenantRateLimiter

LEAK_CANARY = "sk-live-NEXUSLEAKCANARY0123456789"
OTHER_CANARY = "sk-live-NEXUSOTHERCANARY9876543210"
KEY_ID_PATTERN = re.compile(r"^key_[0-9a-f]{16}$")

GatewayFactory = Callable[..., tuple[TestClient, NexusRouter]]


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GatewayFactory:
    """Build an isolated API client backed by mock providers and tmp state files."""

    def _build(**overrides: float) -> tuple[TestClient, NexusRouter]:
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
        )
        router = NexusRouter(settings, registry)
        ledger = SpendLedger(settings.spend_ledger_path)
        cache = ResponseCache(enabled=False)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=60, refill_per_second=1.0),
        )
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_spend_ledger", lambda: ledger)
        monkeypatch.setattr(main, "get_response_cache", lambda: cache)
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        return TestClient(main.app), router

    return _build


def _chat(client: TestClient, prompt: str, api_key: str, **headers: str) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}]},
        headers={"Authorization": f"Bearer {api_key}", **headers},
    )


def test_rate_limit_429_detail_does_not_echo_authorization_secret(
    gateway: GatewayFactory,
) -> None:
    """A rate-limited caller must not get its own API key echoed in the 429 body."""
    client, _ = gateway(rate_limit_capacity=1, rate_limit_refill_per_second=0.0001)
    assert _chat(client, "first prompt", LEAK_CANARY).status_code == 200

    limited = _chat(client, "second prompt", LEAK_CANARY)

    assert limited.status_code == 429
    detail = limited.json()["detail"]
    assert LEAK_CANARY not in detail
    assert "Bearer" not in detail
    assert re.search(r"key_[0-9a-f]{16}", detail)


def test_budget_exhaustion_503_detail_does_not_echo_authorization_secret(
    gateway: GatewayFactory,
) -> None:
    """Budget guardrail rejections surface as 503 and must not leak the key."""
    client, _ = gateway(budget_cap_usd=0.0)

    response = _chat(client, "over budget", LEAK_CANARY)

    assert response.status_code == 503
    assert "budget cap exceeded" in response.json()["detail"]
    assert LEAK_CANARY not in response.json()["detail"]


def test_spend_endpoint_tenants_are_fingerprints_not_raw_authorization(
    gateway: GatewayFactory,
) -> None:
    """``GET /v1/spend`` must expose a key fingerprint, never the raw header."""
    client, _ = gateway()
    assert _chat(client, "record some spend", LEAK_CANARY).status_code == 200

    spend = client.get("/v1/spend")

    assert spend.status_code == 200
    assert LEAK_CANARY not in spend.text
    tenants = list(spend.json()["by_tenant"])
    assert len(tenants) == 1
    assert KEY_ID_PATTERN.match(tenants[0])


def test_router_request_identity_is_stable_per_key_fingerprint(
    gateway: GatewayFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same key maps to the same id, different keys to different ids, never the secret."""
    client, router = gateway()
    captured: list[RouterRequest] = []
    original_complete = router.complete

    async def _spy(request: RouterRequest) -> RouterResponse:
        captured.append(request)
        return await original_complete(request)

    monkeypatch.setattr(router, "complete", _spy)

    assert _chat(client, "prompt one", LEAK_CANARY).status_code == 200
    assert _chat(client, "prompt two", LEAK_CANARY).status_code == 200
    assert _chat(client, "prompt three", OTHER_CANARY).status_code == 200

    first, second, other = captured
    for request in captured:
        dumped = request.model_dump_json()
        assert LEAK_CANARY not in dumped
        assert OTHER_CANARY not in dumped
        assert KEY_ID_PATTERN.match(request.api_key_id)
        assert request.user_id == request.api_key_id
    assert first.api_key_id == second.api_key_id
    assert first.api_key_id != other.api_key_id


def test_authorization_secret_is_never_persisted_to_state_files(
    gateway: GatewayFactory, tmp_path: Path
) -> None:
    """Spend ledger, idempotency store, and audit log must not contain the key."""
    client, _ = gateway()
    response = _chat(client, "persist me", LEAK_CANARY, **{"Idempotency-Key": "idem-leak-1"})
    assert response.status_code == 200
    replay = _chat(client, "persist me", LEAK_CANARY, **{"Idempotency-Key": "idem-leak-1"})
    assert replay.json()["id"] == response.json()["id"]

    state_files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert state_files, "expected spend/idempotency/audit state files"
    for path in state_files:
        assert LEAK_CANARY.encode() not in path.read_bytes(), f"secret persisted in {path.name}"


def test_per_key_rate_limit_isolation_is_preserved(gateway: GatewayFactory) -> None:
    """Exhausting one key's bucket must not throttle a different key."""
    client, _ = gateway(rate_limit_capacity=1, rate_limit_refill_per_second=0.0001)

    assert _chat(client, "a1", LEAK_CANARY).status_code == 200
    assert _chat(client, "a2", LEAK_CANARY).status_code == 429
    assert _chat(client, "b1", OTHER_CANARY).status_code == 200
