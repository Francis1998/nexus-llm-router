"""Tests for RopeScaleMismatchBandAdvisor."""

from __future__ import annotations

import pytest

from safety.rope_scale_mismatch_band import RopeScaleMismatchBandAdvisor


def test_within_band() -> None:
    """Below soft is within."""

    advice = RopeScaleMismatchBandAdvisor().advise(request_id="r1", rope_mismatch_ratio=0.0)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Between soft and hard is soft."""

    advice = RopeScaleMismatchBandAdvisor().advise(request_id="r1", rope_mismatch_ratio=0.35)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """At/above hard is breach."""

    advice = RopeScaleMismatchBandAdvisor().advise(request_id="r1", rope_mismatch_ratio=0.5)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Blank request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        RopeScaleMismatchBandAdvisor().advise(request_id=" ", rope_mismatch_ratio=0.0)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="rope_mismatch_ratio"):
        RopeScaleMismatchBandAdvisor().advise(request_id="r1", rope_mismatch_ratio=-0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        RopeScaleMismatchBandAdvisor().advise(
            request_id="r1",
            rope_mismatch_ratio=0.0,
            soft_limit=10.0,
            hard_limit=1.0,
        )
