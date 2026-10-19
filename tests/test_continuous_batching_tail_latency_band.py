"""Tests for ContinuousBatchingTailLatencyBandAdvisor."""

from __future__ import annotations

import pytest

from safety.continuous_batching_tail_latency_band import ContinuousBatchingTailLatencyBandAdvisor


def test_within_band() -> None:
    """Below soft is within."""

    advice = ContinuousBatchingTailLatencyBandAdvisor().advise(request_id="r1", p99_tail_ms=0.0)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Between soft and hard is soft."""

    advice = ContinuousBatchingTailLatencyBandAdvisor().advise(request_id="r1", p99_tail_ms=525.0)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """At/above hard is breach."""

    advice = ContinuousBatchingTailLatencyBandAdvisor().advise(request_id="r1", p99_tail_ms=800.0)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Blank request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        ContinuousBatchingTailLatencyBandAdvisor().advise(request_id=" ", p99_tail_ms=0.0)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="p99_tail_ms"):
        ContinuousBatchingTailLatencyBandAdvisor().advise(request_id="r1", p99_tail_ms=-0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        ContinuousBatchingTailLatencyBandAdvisor().advise(
            request_id="r1",
            p99_tail_ms=0.0,
            soft_limit=10.0,
            hard_limit=1.0,
        )
