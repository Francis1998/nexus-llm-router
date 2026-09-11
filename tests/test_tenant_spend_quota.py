"""Tests for TenantSpendQuotaEnforcer hard monthly USD caps."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from safety.spend_ledger import SpendLedger
from safety.tenant_spend_quota import (
    QuotaExceededError,
    TenantSpendQuotaEnforcer,
    TenantSpendQuotaSnapshot,
    calendar_month_utc_window,
)


class _Clock:
    """Injectable unix-seconds clock."""

    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _september_2026_mid() -> float:
    return datetime(2026, 9, 15, 12, 0, tzinfo=UTC).timestamp()


def test_calendar_month_utc_window_bounds() -> None:
    """Window is the half-open UTC calendar month containing now."""
    now = _september_2026_mid()
    start, end = calendar_month_utc_window(now)
    assert datetime.fromtimestamp(start, tz=UTC) == datetime(
        2026, 9, 1, tzinfo=UTC
    )
    assert datetime.fromtimestamp(end, tz=UTC) == datetime(
        2026, 10, 1, tzinfo=UTC
    )


def test_assert_within_quota_allows_under_limit(tmp_path: Path) -> None:
    """Tenants under the monthly cap pass the fail-closed gate."""
    ledger = SpendLedger(tmp_path / "under.sqlite3")
    clock = _Clock(_september_2026_mid())
    enforcer = TenantSpendQuotaEnforcer(ledger, monthly_limit_usd=10.0, now_fn=clock)
    ledger.record(
        request_id="r1",
        tenant="acme",
        provider="openai",
        model="gpt-5.5",
        cost_usd=3.5,
        recorded_at=datetime(2026, 9, 10, tzinfo=UTC).timestamp(),
    )
    enforcer.assert_within_quota("acme")
    assert enforcer.remaining("acme") == pytest.approx(6.5)
    snap = enforcer.snapshot("acme")
    assert isinstance(snap, TenantSpendQuotaSnapshot)
    assert snap.exceeded is False
    assert snap.spent_usd == pytest.approx(3.5)


def test_assert_within_quota_raises_when_exhausted(tmp_path: Path) -> None:
    """Spent at or above the monthly limit raises QuotaExceededError."""
    ledger = SpendLedger(tmp_path / "over.sqlite3")
    clock = _Clock(_september_2026_mid())
    enforcer = TenantSpendQuotaEnforcer(ledger, monthly_limit_usd=5.0, now_fn=clock)
    ledger.record(
        request_id="r1",
        tenant="acme",
        provider="anthropic",
        model="claude-sonnet-4-6",
        cost_usd=5.0,
        recorded_at=datetime(2026, 9, 5, tzinfo=UTC).timestamp(),
    )
    with pytest.raises(QuotaExceededError) as exc_info:
        enforcer.assert_within_quota("acme")
    assert exc_info.value.snapshot.exceeded is True
    assert enforcer.remaining("acme") == 0.0


def test_prior_month_spend_excluded_from_quota(tmp_path: Path) -> None:
    """Only the current UTC calendar month counts toward the cap."""
    ledger = SpendLedger(tmp_path / "month.sqlite3")
    clock = _Clock(_september_2026_mid())
    enforcer = TenantSpendQuotaEnforcer(ledger, monthly_limit_usd=1.0, now_fn=clock)
    ledger.record(
        request_id="aug",
        tenant="acme",
        provider="google",
        model="gemini-3.5-flash",
        cost_usd=9.0,
        recorded_at=datetime(2026, 8, 31, 23, 0, tzinfo=UTC).timestamp(),
    )
    ledger.record(
        request_id="sep",
        tenant="acme",
        provider="moonshot",
        model="kimi-k2",
        cost_usd=0.4,
        recorded_at=datetime(2026, 9, 1, 0, 30, tzinfo=UTC).timestamp(),
    )
    enforcer.assert_within_quota("acme")
    assert enforcer.remaining("acme") == pytest.approx(0.6)
    assert enforcer.snapshot("acme").spent_usd == pytest.approx(0.4)


def test_tenants_are_isolated(tmp_path: Path) -> None:
    """One tenant's spend does not exhaust another tenant's quota."""
    ledger = SpendLedger(tmp_path / "iso.sqlite3")
    clock = _Clock(_september_2026_mid())
    enforcer = TenantSpendQuotaEnforcer(ledger, monthly_limit_usd=2.0, now_fn=clock)
    ledger.record(
        request_id="a",
        tenant="acme",
        provider="openai",
        model="gpt-5.5",
        cost_usd=2.0,
        recorded_at=datetime(2026, 9, 2, tzinfo=UTC).timestamp(),
    )
    ledger.record(
        request_id="b",
        tenant="globex",
        provider="anthropic",
        model="claude-sonnet-4-6",
        cost_usd=0.5,
        recorded_at=datetime(2026, 9, 2, tzinfo=UTC).timestamp(),
    )
    with pytest.raises(QuotaExceededError):
        enforcer.assert_within_quota("acme")
    enforcer.assert_within_quota("globex")
    assert enforcer.remaining("globex") == pytest.approx(1.5)


def test_rejects_invalid_limit_and_empty_tenant(tmp_path: Path) -> None:
    """Construction and assert validate limit and tenant."""
    ledger = SpendLedger(tmp_path / "invalid.sqlite3")
    with pytest.raises(ValueError, match="monthly_limit_usd"):
        TenantSpendQuotaEnforcer(ledger, monthly_limit_usd=0)
    with pytest.raises(ValueError, match="monthly_limit_usd"):
        TenantSpendQuotaEnforcer(ledger, monthly_limit_usd=-1.0)
    enforcer = TenantSpendQuotaEnforcer(ledger, monthly_limit_usd=10.0)
    with pytest.raises(ValueError, match="tenant"):
        enforcer.assert_within_quota("")
    with pytest.raises(ValueError, match="tenant"):
        enforcer.remaining("")


def test_frontier_model_spend_counts_toward_monthly_cap(tmp_path: Path) -> None:
    """GPT-5.5 / Sonnet 4.6 / Gemini 3.x / Kimi K2 costs share the tenant cap."""
    ledger = SpendLedger(tmp_path / "frontier.sqlite3")
    clock = _Clock(_september_2026_mid())
    enforcer = TenantSpendQuotaEnforcer(ledger, monthly_limit_usd=1.0, now_fn=clock)
    stamp = datetime(2026, 9, 12, tzinfo=UTC).timestamp()
    for request_id, provider, model, cost in (
        ("g1", "openai", "gpt-5.5", 0.25),
        ("g2", "anthropic", "claude-sonnet-4-6", 0.25),
        ("g3", "google", "gemini-3.5-pro", 0.25),
        ("g4", "moonshot", "kimi-k2", 0.25),
    ):
        ledger.record(
            request_id=request_id,
            tenant="acme",
            provider=provider,
            model=model,
            cost_usd=cost,
            recorded_at=stamp,
        )
    with pytest.raises(QuotaExceededError) as exc_info:
        enforcer.assert_within_quota("acme")
    assert exc_info.value.snapshot.spent_usd == pytest.approx(1.0)
    assert enforcer.monthly_limit_usd == 1.0
