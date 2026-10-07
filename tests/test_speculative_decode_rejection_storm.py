"""Unit tests for SpeculativeDecodeRejectionStormAdvisor."""

from __future__ import annotations

import pytest

from safety.speculative_decode_rejection_storm import SpeculativeDecodeRejectionStormAdvisor


def test_within() -> None:
    """Good metric is within."""

    advice = SpeculativeDecodeRejectionStormAdvisor().advise(request_id="r1", rejection_rate=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = SpeculativeDecodeRejectionStormAdvisor().advise(request_id="r1", rejection_rate=0.45)
    assert advice.band == "soft"


def test_breach() -> None:
    """Bad metric is breach."""

    advice = SpeculativeDecodeRejectionStormAdvisor().advise(request_id="r1", rejection_rate=0.9)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="rejection_rate"):
        SpeculativeDecodeRejectionStormAdvisor().advise(request_id="r1", rejection_rate=-0.1)
