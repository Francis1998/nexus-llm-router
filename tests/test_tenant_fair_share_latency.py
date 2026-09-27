"""Unit tests for TenantFairShareLatencySloAdvisor."""

from __future__ import annotations

import pytest

from safety.tenant_fair_share_latency import TenantFairShareLatencySloAdvisor


def test_ok_band() -> None:
    """Under SLO is ok."""

    advice = TenantFairShareLatencySloAdvisor().advise(
        tenant_id="t1", tenant_p95_ms=80.0, fair_share_slo_ms=100.0
    )
    assert advice.band == "ok"


def test_watch_band() -> None:
    """Slightly over SLO is watch."""

    advice = TenantFairShareLatencySloAdvisor().advise(
        tenant_id="t1", tenant_p95_ms=120.0, fair_share_slo_ms=100.0
    )
    assert advice.band == "watch"


def test_breach_band() -> None:
    """1.5x+ SLO is breach."""

    advice = TenantFairShareLatencySloAdvisor().advise(
        tenant_id="t1", tenant_p95_ms=200.0, fair_share_slo_ms=100.0
    )
    assert advice.band == "breach"


def test_empty_tenant_raises() -> None:
    """Empty tenant id raises ValueError."""

    with pytest.raises(ValueError, match="tenant_id"):
        TenantFairShareLatencySloAdvisor().advise(
            tenant_id=" ", tenant_p95_ms=1.0, fair_share_slo_ms=1.0
        )
