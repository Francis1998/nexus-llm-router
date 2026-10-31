"""Tests for MoeExpertHotspotBandAdvisor."""

from __future__ import annotations

import pytest

from safety.moe_expert_hotspot_band import MoeExpertHotspotBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = MoeExpertHotspotBandAdvisor().advise(request_id="r1", hotspot_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = MoeExpertHotspotBandAdvisor().advise(request_id="r1", hotspot_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = MoeExpertHotspotBandAdvisor().advise(request_id="r1", hotspot_ratio=0.9)
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        MoeExpertHotspotBandAdvisor().advise(request_id=" ", hotspot_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="hotspot_ratio"):
        MoeExpertHotspotBandAdvisor().advise(request_id="r1", hotspot_ratio=-0.1)
