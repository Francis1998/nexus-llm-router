"""Tests for RequestHedgingAdvisor hold/hedge/skip bands."""

from __future__ import annotations

import pytest

from safety.request_hedging import RequestHedgingAdvisor


def test_hold_below_threshold() -> None:
    advice = RequestHedgingAdvisor(slo_ms=1000.0, hedge_ratio=0.75).advise("r1", wait_ms=100.0)
    assert advice.band == "hold"
    assert advice.hedge is False


def test_hedge_in_window() -> None:
    advice = RequestHedgingAdvisor(slo_ms=1000.0, hedge_ratio=0.75).advise("r2", wait_ms=800.0)
    assert advice.band == "hedge"
    assert advice.hedge is True


def test_skip_past_slo() -> None:
    advice = RequestHedgingAdvisor(slo_ms=500.0, hedge_ratio=0.5).advise("r3", wait_ms=500.0)
    assert advice.band == "skip"
    assert advice.hedge is False


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="slo_ms"):
        RequestHedgingAdvisor(slo_ms=0)
    with pytest.raises(ValueError, match="request_id"):
        RequestHedgingAdvisor().advise("", wait_ms=1.0)
    with pytest.raises(ValueError, match="wait_ms"):
        RequestHedgingAdvisor().advise("r", wait_ms=-1.0)


def test_hedge_ratio_bounds() -> None:
    with pytest.raises(ValueError, match="hedge_ratio"):
        RequestHedgingAdvisor(hedge_ratio=1.0)
