"""Unit tests for RegionFailoverHysteresisAdvisor."""

from __future__ import annotations

import pytest

from safety.region_failover_hysteresis import RegionFailoverHysteresisAdvisor


def test_within() -> None:
    """Band within."""

    advice = RegionFailoverHysteresisAdvisor().advise(request_id="r1", failover_score=0.15)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = RegionFailoverHysteresisAdvisor().advise(request_id="r1", failover_score=0.5)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = RegionFailoverHysteresisAdvisor().advise(request_id="r1", failover_score=1.7)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="failover_score"):
        RegionFailoverHysteresisAdvisor().advise(request_id="r1", failover_score=-0.1)
