"""Unit tests for SpeculativeDecodeAcceptanceBandAdvisor."""

from __future__ import annotations

import pytest

from safety.speculative_decode_acceptance import SpeculativeDecodeAcceptanceBandAdvisor


def test_within() -> None:
    """High acceptance is within."""

    advice = SpeculativeDecodeAcceptanceBandAdvisor().advise(request_id="r1", acceptance_rate=0.9)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid acceptance is soft."""

    advice = SpeculativeDecodeAcceptanceBandAdvisor().advise(request_id="r1", acceptance_rate=0.55)
    assert advice.band == "soft"


def test_breach() -> None:
    """Low acceptance is breach."""

    advice = SpeculativeDecodeAcceptanceBandAdvisor().advise(request_id="r1", acceptance_rate=0.2)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Empty request id raises."""

    with pytest.raises(ValueError, match="request_id"):
        SpeculativeDecodeAcceptanceBandAdvisor().advise(request_id="  ", acceptance_rate=0.9)
