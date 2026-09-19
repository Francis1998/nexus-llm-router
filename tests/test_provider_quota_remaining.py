"""Tests for ProviderQuotaRemainingAdvisor ok/low/exhausted bands."""

from __future__ import annotations

import pytest

from safety.provider_quota_remaining import ProviderQuotaRemainingAdvisor


def test_ok_band() -> None:
    adv = ProviderQuotaRemainingAdvisor(low_ratio=0.2).observe(
        "openai", remaining=800, capacity=1000
    )
    assert adv.band == "ok"
    assert adv.remaining_ratio == 0.8


def test_low_band() -> None:
    adv = ProviderQuotaRemainingAdvisor(low_ratio=0.2).observe(
        "anthropic", remaining=100, capacity=1000
    )
    assert adv.band == "low"


def test_exhausted_band() -> None:
    adv = ProviderQuotaRemainingAdvisor().observe("gemini", remaining=0, capacity=500)
    assert adv.band == "exhausted"
    assert adv.remaining == 0


def test_snapshot_roundtrip() -> None:
    advisor = ProviderQuotaRemainingAdvisor()
    advisor.observe("kimi", remaining=50, capacity=100)
    snap = advisor.snapshot("kimi")
    assert snap is not None
    assert snap.band == "ok"


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="capacity"):
        ProviderQuotaRemainingAdvisor().observe("x", remaining=1, capacity=0)
    with pytest.raises(ValueError, match="provider"):
        ProviderQuotaRemainingAdvisor().observe("", remaining=1, capacity=1)
