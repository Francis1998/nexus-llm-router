"""Tests for SpeculativeDecodeMismatchFloorAdvisor."""

from __future__ import annotations

import pytest

from safety.speculative_decode_mismatch_floor import SpeculativeDecodeMismatchFloorAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = SpeculativeDecodeMismatchFloorAdvisor().advise(request_id="r1", mismatch_ratio=0.01)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = SpeculativeDecodeMismatchFloorAdvisor().advise(
        request_id="r1",
        mismatch_ratio=0.2,
        soft_limit=0.2,
        hard_limit=0.55,
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = SpeculativeDecodeMismatchFloorAdvisor().advise(
        request_id="r1",
        mismatch_ratio=0.55,
        soft_limit=0.2,
        hard_limit=0.55,
    )
    assert advice.band == "breach"


def test_invalid() -> None:
    """Invalid inputs raise."""

    with pytest.raises(ValueError, match="request_id"):
        SpeculativeDecodeMismatchFloorAdvisor().advise(request_id=" ", mismatch_ratio=0.1)
