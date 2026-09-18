"""Tests for RequestPriorityAgingAdvisor fresh/aging/stale boost bands."""

from __future__ import annotations

import pytest

from safety.priority_aging import PriorityAgingAdvice, RequestPriorityAgingAdvisor


def test_fresh_band_when_under_aging_threshold() -> None:
    """Young wait ages yield fresh with zero boost."""
    advisor = RequestPriorityAgingAdvisor(aging_after_seconds=5.0, stale_after_seconds=30.0)
    advice = advisor.advise(age_seconds=1.0, request_id="gpt-5.5-req")
    assert isinstance(advice, PriorityAgingAdvice)
    assert advice.band == "fresh"
    assert advice.boost == 0
    assert "fresh" in advice.advisory.lower()
    assert advice.request_id == "gpt-5.5-req"


def test_aging_band_at_aging_threshold() -> None:
    """Age at/above aging_after but under stale_after is aging."""
    advisor = RequestPriorityAgingAdvisor(
        aging_after_seconds=5.0,
        stale_after_seconds=30.0,
        aging_boost=1,
        stale_boost=3,
    )
    advice = advisor.advise(age_seconds=5.0, request_id="claude-sonnet-4.6-req")
    assert advice.band == "aging"
    assert advice.boost == 1
    assert "aging" in advice.advisory.lower()


def test_stale_band_at_stale_threshold() -> None:
    """Age at/above stale_after yields stale with the stale boost."""
    advisor = RequestPriorityAgingAdvisor(
        aging_after_seconds=5.0,
        stale_after_seconds=30.0,
        aging_boost=1,
        stale_boost=3,
    )
    advice = advisor.advise(age_seconds=30.0, request_id="gemini-3-req")
    assert advice.band == "stale"
    assert advice.boost == 3
    assert "stale" in advice.advisory.lower()


def test_frontier_models_share_aging_thresholds() -> None:
    """Same thresholds apply across GPT-5.5 / Sonnet / Gemini / Kimi request ids."""
    advisor = RequestPriorityAgingAdvisor(aging_after_seconds=10.0, stale_after_seconds=60.0)
    for request_id in (
        "gpt-5.5",
        "claude-sonnet-4.6",
        "gemini-3.1-pro-preview",
        "kimi-k2",
    ):
        advice = advisor.advise(age_seconds=12.0, request_id=request_id)
        assert advice.band == "aging"
        assert advice.request_id == request_id


def test_rejects_invalid_inputs() -> None:
    """Constructor and advise validate thresholds, boosts, and age_seconds."""
    with pytest.raises(ValueError, match="aging_after_seconds"):
        RequestPriorityAgingAdvisor(aging_after_seconds=0.0, stale_after_seconds=10.0)
    with pytest.raises(ValueError, match="stale_after_seconds"):
        RequestPriorityAgingAdvisor(aging_after_seconds=10.0, stale_after_seconds=10.0)
    with pytest.raises(ValueError, match="aging_boost"):
        RequestPriorityAgingAdvisor(aging_boost=-1)
    with pytest.raises(ValueError, match="stale_boost"):
        RequestPriorityAgingAdvisor(aging_boost=2, stale_boost=1)
    advisor = RequestPriorityAgingAdvisor()
    with pytest.raises(ValueError, match="age_seconds"):
        advisor.advise(age_seconds=-0.1)
    # Distinct from RequestPriorityLane — advisor returns boost, never dequeues.
    assert advisor.advise(age_seconds=0.0).band == "fresh"
