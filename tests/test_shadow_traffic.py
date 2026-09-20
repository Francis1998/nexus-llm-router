"""Tests for ShadowTrafficMirrorGuard off/mirror/full bands."""

from __future__ import annotations

import pytest

from safety.shadow_traffic import ShadowTrafficMirrorGuard


def test_off_when_zero() -> None:
    decision = ShadowTrafficMirrorGuard(mirror_pct=0.0).decide("r1", sample_unit=0.0)
    assert decision.band == "off"
    assert decision.should_mirror is False


def test_full_when_hundred() -> None:
    decision = ShadowTrafficMirrorGuard(mirror_pct=100.0).decide("r2", sample_unit=0.99)
    assert decision.band == "full"
    assert decision.should_mirror is True


def test_mirror_sampling() -> None:
    guard = ShadowTrafficMirrorGuard(mirror_pct=10.0)
    yes = guard.decide("r3", sample_unit=0.05)
    no = guard.decide("r4", sample_unit=0.50)
    assert yes.band == "mirror"
    assert yes.should_mirror is True
    assert no.should_mirror is False


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="mirror_pct"):
        ShadowTrafficMirrorGuard(mirror_pct=101.0)
    with pytest.raises(ValueError, match="request_id"):
        ShadowTrafficMirrorGuard().decide("", sample_unit=0.1)
    with pytest.raises(ValueError, match="sample_unit"):
        ShadowTrafficMirrorGuard().decide("r", sample_unit=1.0)


def test_mid_pct_property() -> None:
    assert ShadowTrafficMirrorGuard(mirror_pct=5.0).mirror_pct == 5.0
