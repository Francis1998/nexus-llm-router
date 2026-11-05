"""Tests for ContinuousBatchPadWasteBandAdvisor."""

from __future__ import annotations

import pytest

from safety.continuous_batch_pad_waste_band import ContinuousBatchPadWasteBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = ContinuousBatchPadWasteBandAdvisor().advise(
        request_id="r1", pad_waste_ratio=0.1, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = ContinuousBatchPadWasteBandAdvisor().advise(
        request_id="r1", pad_waste_ratio=0.3, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = ContinuousBatchPadWasteBandAdvisor().advise(
        request_id="r1", pad_waste_ratio=0.9, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        ContinuousBatchPadWasteBandAdvisor().advise(request_id=" ", pad_waste_ratio=0.1)


def test_bad_limits() -> None:
    """Invalid limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        ContinuousBatchPadWasteBandAdvisor().advise(
            request_id="r1", pad_waste_ratio=0.1, soft_limit=0.5, hard_limit=0.2
        )
