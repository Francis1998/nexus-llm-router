"""Tests for StreamingTtftSpikeBandAdvisor."""

from __future__ import annotations

import pytest

from safety.streaming_ttft_spike_band import StreamingTtftSpikeBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = StreamingTtftSpikeBandAdvisor().advise(
        request_id="r1", spike_ratio=0.1, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = StreamingTtftSpikeBandAdvisor().advise(
        request_id="r1", spike_ratio=0.3, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = StreamingTtftSpikeBandAdvisor().advise(
        request_id="r1", spike_ratio=0.9, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        StreamingTtftSpikeBandAdvisor().advise(request_id=" ", spike_ratio=0.1)


def test_bad_limits() -> None:
    """Invalid limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        StreamingTtftSpikeBandAdvisor().advise(
            request_id="r1", spike_ratio=0.1, soft_limit=0.5, hard_limit=0.2
        )
