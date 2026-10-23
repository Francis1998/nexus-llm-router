"""Unit tests for DraftVerifyMismatchBandAdvisor."""

from __future__ import annotations

import pytest

from safety.draft_verify_mismatch_band import (
    DraftVerifyMismatchBandAdvisor,
)


def test_within() -> None:
    """Below soft -> within."""

    advice = DraftVerifyMismatchBandAdvisor().advise(request_id="r1", mismatch_ratio=0.05)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard -> soft."""

    advice = DraftVerifyMismatchBandAdvisor().advise(request_id="r1", mismatch_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard -> breach."""

    advice = DraftVerifyMismatchBandAdvisor().advise(request_id="r1", mismatch_ratio=0.8)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        DraftVerifyMismatchBandAdvisor().advise(request_id=" ", mismatch_ratio=0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        DraftVerifyMismatchBandAdvisor().advise(
            request_id="r1",
            mismatch_ratio=0.1,
            soft_limit=0.5,
            hard_limit=0.2,
        )
