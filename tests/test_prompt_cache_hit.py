"""Tests for PromptCacheHitRateAdvisor cold/warm/hot bands."""

from __future__ import annotations

import pytest

from safety.prompt_cache_hit import PromptCacheHitAdvice, PromptCacheHitRateAdvisor


def test_cold_band_when_hit_rate_below_warm() -> None:
    """Low cached_tokens / prompt_tokens yields cold advisory."""
    advisor = PromptCacheHitRateAdvisor(warm_min_hit_rate=0.25, hot_min_hit_rate=0.75)
    advice = advisor.record(cached_tokens=100, prompt_tokens=10_000, model="gpt-5.5")
    assert isinstance(advice, PromptCacheHitAdvice)
    assert advice.band == "cold"
    assert advice.hit_rate == pytest.approx(0.01)
    assert "cold" in advice.advisory.lower()
    assert advice.model == "gpt-5.5"


def test_warm_band_between_thresholds() -> None:
    """Hit rate at/above warm_min but under hot_min is warm."""
    advisor = PromptCacheHitRateAdvisor(warm_min_hit_rate=0.25, hot_min_hit_rate=0.75)
    advice = advisor.record(
        cached_tokens=4_000,
        prompt_tokens=10_000,
        model="claude-sonnet-4.6",
    )
    assert advice.band == "warm"
    assert advice.hit_rate == pytest.approx(0.4)
    assert "warm" in advice.advisory.lower()


def test_hot_band_at_or_above_hot_min() -> None:
    """Hit rate at/above hot_min yields hot."""
    advisor = PromptCacheHitRateAdvisor()
    advice = advisor.record(
        cached_tokens=9_000,
        prompt_tokens=10_000,
        model="gemini-3.1-pro-preview",
    )
    assert advice.band == "hot"
    assert advice.hit_rate == pytest.approx(0.9)
    assert "hot" in advice.advisory.lower()


def test_frontier_models_share_thresholds_and_rolling() -> None:
    """Same thresholds apply across GPT-5.5 / Sonnet / Gemini / Kimi."""
    advisor = PromptCacheHitRateAdvisor(warm_min_hit_rate=0.3, hot_min_hit_rate=0.8)
    for model in (
        "gpt-5.5",
        "claude-sonnet-4.6",
        "gemini-3.1-pro-preview",
        "kimi-k2",
    ):
        advice = advisor.record(cached_tokens=5_000, prompt_tokens=10_000, model=model)
        assert advice.band == "warm"
        assert advice.model == model
        assert advice.samples >= 1
        assert advice.rolling_hit_rate is not None

    snap = advisor.snapshot()
    assert snap["samples"] == 4
    assert snap["rolling_hit_rate"] == pytest.approx(0.5)
    advisor.clear()
    assert advisor.snapshot()["samples"] == 0


def test_rejects_invalid_inputs() -> None:
    """Constructor and record validate thresholds and token counts."""
    with pytest.raises(ValueError, match="warm_min_hit_rate"):
        PromptCacheHitRateAdvisor(warm_min_hit_rate=0.0)
    with pytest.raises(ValueError, match="hot_min_hit_rate"):
        PromptCacheHitRateAdvisor(hot_min_hit_rate=0.1, warm_min_hit_rate=0.25)
    with pytest.raises(ValueError, match="window_size"):
        PromptCacheHitRateAdvisor(window_size=0)
    advisor = PromptCacheHitRateAdvisor()
    with pytest.raises(ValueError, match="cached_tokens"):
        advisor.record(cached_tokens=-1, prompt_tokens=10)
    with pytest.raises(ValueError, match="prompt_tokens"):
        advisor.record(cached_tokens=1, prompt_tokens=-1)
    with pytest.raises(ValueError, match="cached_tokens must be <="):
        advisor.record(cached_tokens=20, prompt_tokens=10)
    zero = advisor.record(cached_tokens=0, prompt_tokens=0)
    assert zero.hit_rate == 0.0
    assert zero.band == "cold"
