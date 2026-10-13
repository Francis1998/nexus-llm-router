"""Tests for AdapterWeightDriftAdvisor."""

from __future__ import annotations

import pytest

from safety.adapter_weight_drift import AdapterWeightDriftAdvisor


def test_within() -> None:
    """Band within."""

    advice = AdapterWeightDriftAdvisor().advise(request_id="r1", weight_drift_score=0.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.2 + 0.5) / 2
    advice = AdapterWeightDriftAdvisor().advise(request_id="r1", weight_drift_score=mid)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = AdapterWeightDriftAdvisor().advise(request_id="r1", weight_drift_score=1.0000)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError, match="request_id"):
        AdapterWeightDriftAdvisor().advise(request_id=" ", weight_drift_score=0.0)
