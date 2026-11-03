"""Tests for TensorParallelAllreduceStallBandAdvisor."""

from __future__ import annotations

import pytest

from safety.tensor_parallel_allreduce_stall_band import TensorParallelAllreduceStallBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = TensorParallelAllreduceStallBandAdvisor().advise(
        request_id="r1", stall_ratio=0.1, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = TensorParallelAllreduceStallBandAdvisor().advise(
        request_id="r1", stall_ratio=0.3, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = TensorParallelAllreduceStallBandAdvisor().advise(
        request_id="r1", stall_ratio=0.6, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        TensorParallelAllreduceStallBandAdvisor().advise(request_id=" ", stall_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="stall_ratio"):
        TensorParallelAllreduceStallBandAdvisor().advise(request_id="r1", stall_ratio=-0.1)
