"""Tests for SpeculativeVerifyBacklogBandAdvisor."""

from __future__ import annotations

import pytest

from safety.speculative_verify_backlog_band import SpeculativeVerifyBacklogBandAdvisor


def test_within() -> None:
    """Within soft."""

    advice = SpeculativeVerifyBacklogBandAdvisor().advise(request_id="r1", backlog_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Soft band."""

    advice = SpeculativeVerifyBacklogBandAdvisor().advise(request_id="r1", backlog_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """Breach band."""

    advice = SpeculativeVerifyBacklogBandAdvisor().advise(request_id="r1", backlog_ratio=0.9)
    assert advice.band == "breach"


def test_invalid_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        SpeculativeVerifyBacklogBandAdvisor().advise(request_id=" ", backlog_ratio=0.1)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="backlog_ratio"):
        SpeculativeVerifyBacklogBandAdvisor().advise(request_id="r1", backlog_ratio=-0.1)
