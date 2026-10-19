"""Tests for SpeculativeTokenVerifyLagAdvisor."""

from __future__ import annotations

import pytest

from safety.speculative_token_verify_lag import SpeculativeTokenVerifyLagAdvisor


def test_within_band() -> None:
    """Below soft is within."""

    advice = SpeculativeTokenVerifyLagAdvisor().advise(request_id="r1", verify_lag_ms=0.0)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Between soft and hard is soft."""

    advice = SpeculativeTokenVerifyLagAdvisor().advise(request_id="r1", verify_lag_ms=80.0)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """At/above hard is breach."""

    advice = SpeculativeTokenVerifyLagAdvisor().advise(request_id="r1", verify_lag_ms=120.0)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Blank request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        SpeculativeTokenVerifyLagAdvisor().advise(request_id=" ", verify_lag_ms=0.0)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="verify_lag_ms"):
        SpeculativeTokenVerifyLagAdvisor().advise(request_id="r1", verify_lag_ms=-0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        SpeculativeTokenVerifyLagAdvisor().advise(
            request_id="r1",
            verify_lag_ms=0.0,
            soft_limit=10.0,
            hard_limit=1.0,
        )
