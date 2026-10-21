"""Tests for QuantCalibrationDriftBandAdvisor."""

from __future__ import annotations

import pytest

from safety.quant_calibration_drift_band import QuantCalibrationDriftBandAdvisor


def test_within_band() -> None:
    """Below soft is within."""

    advice = QuantCalibrationDriftBandAdvisor().advise(request_id="r1", calib_drift_ratio=0.0)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Between soft and hard is soft."""

    advice = QuantCalibrationDriftBandAdvisor().advise(request_id="r1", calib_drift_ratio=0.35)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """At/above hard is breach."""

    advice = QuantCalibrationDriftBandAdvisor().advise(request_id="r1", calib_drift_ratio=0.5)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Blank request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        QuantCalibrationDriftBandAdvisor().advise(request_id=" ", calib_drift_ratio=0.0)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="calib_drift_ratio"):
        QuantCalibrationDriftBandAdvisor().advise(request_id="r1", calib_drift_ratio=-0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        QuantCalibrationDriftBandAdvisor().advise(
            request_id="r1",
            calib_drift_ratio=0.0,
            soft_limit=10.0,
            hard_limit=1.0,
        )
