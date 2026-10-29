"""Tests for ExpertParallelImbalanceBandAdvisor."""

from __future__ import annotations

import pytest

from safety.expert_parallel_imbalance_band import ExpertParallelImbalanceBandAdvisor


def test_within() -> None:
    """Within soft."""

    advice = ExpertParallelImbalanceBandAdvisor().advise(request_id="r1", imbalance_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Soft band."""

    advice = ExpertParallelImbalanceBandAdvisor().advise(request_id="r1", imbalance_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """Breach band."""

    advice = ExpertParallelImbalanceBandAdvisor().advise(request_id="r1", imbalance_ratio=0.9)
    assert advice.band == "breach"


def test_invalid_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        ExpertParallelImbalanceBandAdvisor().advise(request_id=" ", imbalance_ratio=0.1)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="imbalance_ratio"):
        ExpertParallelImbalanceBandAdvisor().advise(request_id="r1", imbalance_ratio=-0.1)
