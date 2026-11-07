"""OpenAI-compatible model discovery: GET /v1/models and GET /v1/models/{id}."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api.main as main
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings, default_model_catalog
from router.engine import NexusRouter
from router.model_ids import (
    ANTHROPIC_FAST_MODEL,
    ANTHROPIC_SAFETY_MODEL,
    OPENAI_FRONTIER_MODEL,
)
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.tenant_rate_limiter import TenantRateLimiter

PROVIDERS = ("openai", "anthropic", "google", "moonshot")

ClientFactory = Callable[..., TestClient]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ClientFactory:
    """Build an API client over mock providers; ``failing`` providers report unhealthy."""

    def _build(*, failing: tuple[str, ...] = ()) -> TestClient:
        settings = RouterSettings(
            audit_log_path=str(tmp_path / "audit-log.jsonl"),
            spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
            idempotency_path=str(tmp_path / "idempotency.sqlite3"),
            response_cache_enabled=False,
        )
        registry = AdapterRegistry(
            {name: MockProviderAdapter(name, fail=name in failing) for name in PROVIDERS}
        )
        router = NexusRouter(settings, registry)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=60, refill_per_second=1.0),
        )
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: ResponseCache(ttl_seconds=60.0))
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        return TestClient(main.app)

    return _build


def _by_id(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {model["id"]: model for model in body["data"]}


def test_list_models_is_openai_compatible(client: ClientFactory) -> None:
    """The list has the OpenAI shape and every routable catalog model."""
    response = client().get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    catalog = default_model_catalog()
    assert [model["id"] for model in body["data"]] == list(catalog)
    for model in body["data"]:
        assert model["object"] == "model"
        assert isinstance(model["created"], int)
        assert model["owned_by"] == catalog[model["id"]].provider


def test_models_expose_nexus_pricing_limits_and_capabilities(client: ClientFactory) -> None:
    """The nexus block carries per-1M prices, long-prompt tiers, input limits, capabilities."""
    models = _by_id(client().get("/v1/models").json())

    astra = models[OPENAI_FRONTIER_MODEL]["nexus"]
    assert astra["provider"] == "openai"
    assert astra["context_window"] == 1_050_000
    assert astra["max_input_tokens"] == 922_000
    assert astra["pricing"] == {
        "input_per_1m_tokens": 10.0,
        "output_per_1m_tokens": 50.0,
        "long_prompt": {
            "threshold_tokens": 272_000,
            "input_per_1m_tokens": 20.0,
            "output_per_1m_tokens": 75.0,
        },
    }
    assert "vision" in astra["capabilities"] and "audio" not in astra["capabilities"]
    assert astra["capabilities"] == sorted(astra["capabilities"])
    assert astra["domains"] == ["code", "general", "legal"]

    haiku = models[ANTHROPIC_FAST_MODEL]["nexus"]
    assert haiku["pricing"]["long_prompt"] == {
        "threshold_tokens": 100_000,
        "input_per_1m_tokens": 0.5,
        "output_per_1m_tokens": 2.5,
    }
    opus = models[ANTHROPIC_SAFETY_MODEL]["nexus"]
    assert opus["pricing"]["long_prompt"] is None
    assert opus["max_input_tokens"] is None


def test_retrieve_model_matches_list_entry(client: ClientFactory) -> None:
    """``GET /v1/models/{id}`` returns the same object as the list entry."""
    api = client()

    listed = _by_id(api.get("/v1/models").json())["gemini-3.8-flash"]
    retrieved = api.get("/v1/models/gemini-3.8-flash")

    assert retrieved.status_code == 200
    assert retrieved.json() == listed


def test_retrieve_unknown_model_is_openai_404(client: ClientFactory) -> None:
    """Unknown ids return the OpenAI error envelope with ``model_not_found``."""
    response = client().get("/v1/models/gpt-4o")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "The model 'gpt-4o' does not exist",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


def test_model_health_reflects_credentials_and_open_circuits(client: ClientFactory) -> None:
    """Unhealthy adapters and open circuit breakers show up per model."""
    api = client(failing=("anthropic",))

    healthy = _by_id(api.get("/v1/models").json())
    assert healthy[OPENAI_FRONTIER_MODEL]["nexus"]["health"] == {
        "api_key_configured": True,
        "routable": True,
    }
    assert healthy[ANTHROPIC_SAFETY_MODEL]["nexus"]["health"]["api_key_configured"] is False

    # Medical prompts route to Anthropic first; three failures open its circuit.
    for _ in range(3):
        api.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Assess patient symptoms."}]},
        )
    tripped = _by_id(api.get("/v1/models").json())
    assert tripped[ANTHROPIC_SAFETY_MODEL]["nexus"]["health"]["routable"] is False
    assert tripped[OPENAI_FRONTIER_MODEL]["nexus"]["health"]["routable"] is True
