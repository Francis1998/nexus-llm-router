"""Unit tests for TensorParallelSkewBandAdvisor."""

from __future__ import annotations

import pytest

from safety.tensor_parallel_skew_band import (
    TensorParallelSkewBandAdvisor,
)


def test_within() -> None:
    """Below soft -> within."""

    advice = TensorParallelSkewBandAdvisor().advise(request_id="r1", skew_ratio=0.1000)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard -> soft."""

    advice = TensorParallelSkewBandAdvisor().advise(request_id="r1", skew_ratio=0.3500)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard -> breach."""

    advice = TensorParallelSkewBandAdvisor().advise(request_id="r1", skew_ratio=0.7000)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        TensorParallelSkewBandAdvisor().advise(request_id=" ", skew_ratio=0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        TensorParallelSkewBandAdvisor().advise(
            request_id="r1",
            skew_ratio=0.1,
            soft_limit=0.5,
            hard_limit=0.2,
        )
