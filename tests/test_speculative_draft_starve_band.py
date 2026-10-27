"""Unit tests for SpeculativeDraftStarveBandAdvisor."""

from __future__ import annotations

import pytest

from safety.speculative_draft_starve_band import (
    SpeculativeDraftStarveBandAdvisor,
)


def test_within() -> None:
    """Below soft -> within."""

    advice = SpeculativeDraftStarveBandAdvisor().advise(request_id="r1", starve_ratio=0.1000)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard -> soft."""

    advice = SpeculativeDraftStarveBandAdvisor().advise(request_id="r1", starve_ratio=0.3500)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard -> breach."""

    advice = SpeculativeDraftStarveBandAdvisor().advise(request_id="r1", starve_ratio=0.7000)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        SpeculativeDraftStarveBandAdvisor().advise(request_id=" ", starve_ratio=0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        SpeculativeDraftStarveBandAdvisor().advise(
            request_id="r1",
            starve_ratio=0.1,
            soft_limit=0.5,
            hard_limit=0.2,
        )
