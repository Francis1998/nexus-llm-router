"""Unit tests for SpeculativeDecodeQueueStallAdvisor."""

from __future__ import annotations

import pytest

from safety.speculative_decode_queue_stall import SpeculativeDecodeQueueStallAdvisor


def test_within() -> None:
    """Band within."""

    advice = SpeculativeDecodeQueueStallAdvisor().advise(request_id="r1", stall_ratio=0.15)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = SpeculativeDecodeQueueStallAdvisor().advise(request_id="r1", stall_ratio=0.5)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = SpeculativeDecodeQueueStallAdvisor().advise(request_id="r1", stall_ratio=1.7)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="stall_ratio"):
        SpeculativeDecodeQueueStallAdvisor().advise(request_id="r1", stall_ratio=-0.1)
