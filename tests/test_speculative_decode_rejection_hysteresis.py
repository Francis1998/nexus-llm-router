"""Unit tests for SpeculativeDecodeRejectionHysteresisAdvisor."""

from __future__ import annotations

import pytest

from safety.speculative_decode_rejection_hysteresis import (
    SpeculativeDecodeRejectionHysteresisAdvisor,
)


def test_within() -> None:
    """Below soft -> within."""

    advice = SpeculativeDecodeRejectionHysteresisAdvisor().advise(
        request_id="r1", hysteresis_ratio=0.05
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard -> soft."""

    advice = SpeculativeDecodeRejectionHysteresisAdvisor().advise(
        request_id="r1", hysteresis_ratio=0.3
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard -> breach."""

    advice = SpeculativeDecodeRejectionHysteresisAdvisor().advise(
        request_id="r1", hysteresis_ratio=0.8
    )
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        SpeculativeDecodeRejectionHysteresisAdvisor().advise(request_id=" ", hysteresis_ratio=0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        SpeculativeDecodeRejectionHysteresisAdvisor().advise(
            request_id="r1",
            hysteresis_ratio=0.1,
            soft_limit=0.5,
            hard_limit=0.2,
        )
