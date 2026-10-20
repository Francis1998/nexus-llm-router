"""Tests for ContinuousBatchPreemptionChurnAdvisor."""

from __future__ import annotations

import pytest

from safety.continuous_batch_preemption_churn import ContinuousBatchPreemptionChurnAdvisor


def test_within_band() -> None:
    """Below soft is within."""

    advice = ContinuousBatchPreemptionChurnAdvisor().advise(
        request_id="r1", preemption_churn_rate=0.0
    )
    assert advice.band == "within"


def test_soft_band() -> None:
    """Between soft and hard is soft."""

    advice = ContinuousBatchPreemptionChurnAdvisor().advise(
        request_id="r1", preemption_churn_rate=0.425
    )
    assert advice.band == "soft"


def test_breach_band() -> None:
    """At/above hard is breach."""

    advice = ContinuousBatchPreemptionChurnAdvisor().advise(
        request_id="r1", preemption_churn_rate=0.6
    )
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Blank request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        ContinuousBatchPreemptionChurnAdvisor().advise(request_id=" ", preemption_churn_rate=0.0)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="preemption_churn_rate"):
        ContinuousBatchPreemptionChurnAdvisor().advise(request_id="r1", preemption_churn_rate=-0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        ContinuousBatchPreemptionChurnAdvisor().advise(
            request_id="r1",
            preemption_churn_rate=0.0,
            soft_limit=10.0,
            hard_limit=1.0,
        )
