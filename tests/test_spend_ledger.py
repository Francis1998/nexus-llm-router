"""Tests for the durable SQLite spend ledger and /v1/spend API."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from api.main import app, get_settings, get_spend_ledger
from safety.spend_ledger import SpendLedger


def test_spend_ledger_persists_across_instances(tmp_path: Path) -> None:
    """A new SpendLedger on the same DB path should see prior records."""
    db_path = tmp_path / "spend.sqlite3"
    first = SpendLedger(db_path)
    first.record(
        request_id="req-1",
        tenant="tenant-a",
        provider="openai",
        model="gpt-5.5",
        cost_usd=0.12,
        input_tokens=100,
        output_tokens=50,
    )
    second = SpendLedger(db_path)
    summary = second.summary()
    assert summary.request_count == 1
    assert summary.total_cost_usd == 0.12
    assert summary.by_tenant["tenant-a"] == 0.12
    assert summary.by_provider["openai"] == 0.12
    assert summary.by_model["gpt-5.5"] == 0.12


def test_spend_ledger_aggregates_by_tenant_provider_model(tmp_path: Path) -> None:
    """Summary aggregates costs across tenants, providers, and models."""
    ledger = SpendLedger(tmp_path / "agg.sqlite3")
    ledger.record(
        request_id="r1",
        tenant="acme",
        provider="anthropic",
        model="claude-sonnet-4-6",
        cost_usd=0.20,
    )
    ledger.record(
        request_id="r2",
        tenant="acme",
        provider="google",
        model="gemini-3.5-flash",
        cost_usd=0.05,
    )
    ledger.record(
        request_id="r3",
        tenant="globex",
        provider="moonshot",
        model="kimi-k2",
        cost_usd=0.07,
    )
    all_summary = ledger.summary()
    assert all_summary.request_count == 3
    assert round(all_summary.total_cost_usd, 2) == 0.32
    assert round(all_summary.by_tenant["acme"], 2) == 0.25
    assert all_summary.by_provider["anthropic"] == 0.20
    acme_only = ledger.summary(tenant="acme")
    assert acme_only.request_count == 2
    assert round(acme_only.total_cost_usd, 2) == 0.25


def test_spend_api_post_and_get(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """POST /v1/spend records an event; GET /v1/spend returns the aggregate."""
    db_path = tmp_path / "api-spend.sqlite3"
    get_settings.cache_clear()
    get_spend_ledger.cache_clear()

    import api.main as main
    from router.config import load_settings

    settings = load_settings()
    object.__setattr__(settings, "spend_ledger_path", str(db_path)) if False else None
    # RouterSettings may be frozen; patch getter instead.
    ledger = SpendLedger(db_path)

    def _settings() -> object:
        return settings

    def _ledger() -> SpendLedger:
        return ledger

    monkeypatch.setattr(main, "get_settings", _settings)
    monkeypatch.setattr(main, "get_spend_ledger", _ledger)

    client = TestClient(app)
    post = client.post(
        "/v1/spend",
        json={
            "request_id": "api-1",
            "tenant": "demo",
            "provider": "openai",
            "model": "gpt-5.5",
            "cost_usd": 0.42,
            "input_tokens": 10,
            "output_tokens": 20,
        },
    )
    assert post.status_code == 200
    body = post.json()
    assert body["request_count"] == 1
    assert body["total_cost_usd"] == 0.42
    assert body["by_model"]["gpt-5.5"] == 0.42

    get = client.get("/v1/spend", params={"tenant": "demo"})
    assert get.status_code == 200
    assert get.json()["total_cost_usd"] == 0.42
