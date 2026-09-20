"""Tests for ProviderCanaryRolloutGuard off/canary/promote bands."""

from __future__ import annotations

import pytest

from safety.provider_canary import ProviderCanaryRolloutGuard


def test_off_when_zero() -> None:
    decision = ProviderCanaryRolloutGuard(canary_pct=0.0).decide("r1", sample_unit=0.0)
    assert decision.band == "off"
    assert decision.use_canary is False


def test_promote_when_hundred() -> None:
    decision = ProviderCanaryRolloutGuard(canary_pct=100.0, provider="gemini").decide(
        "r2", sample_unit=0.5
    )
    assert decision.band == "promote"
    assert decision.use_canary is True
    assert decision.provider == "gemini"


def test_canary_sampling() -> None:
    guard = ProviderCanaryRolloutGuard(canary_pct=20.0, provider="kimi")
    yes = guard.decide("r3", sample_unit=0.10)
    no = guard.decide("r4", sample_unit=0.50)
    assert yes.band == "canary"
    assert yes.use_canary is True
    assert no.use_canary is False


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="canary_pct"):
        ProviderCanaryRolloutGuard(canary_pct=-1.0)
    with pytest.raises(ValueError, match="provider"):
        ProviderCanaryRolloutGuard(provider="  ")
    with pytest.raises(ValueError, match="request_id"):
        ProviderCanaryRolloutGuard().decide("", sample_unit=0.1)


def test_pct_property() -> None:
    assert ProviderCanaryRolloutGuard(canary_pct=7.5).canary_pct == 7.5
