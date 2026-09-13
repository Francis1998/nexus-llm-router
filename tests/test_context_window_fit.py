"""Tests for ContextWindowFitAdvisor estimated-token fit bands."""

from __future__ import annotations

import pytest

from safety.context_window_fit import ContextWindowFitAdvice, ContextWindowFitAdvisor


def test_fits_band_when_well_under_limit() -> None:
    """Low utilization yields fits with positive headroom advisory."""
    advisor = ContextWindowFitAdvisor(tight_ratio=0.85)
    advice = advisor.advise(estimated_tokens=4_000, model_context_limit=128_000, model="gpt-5.5")
    assert isinstance(advice, ContextWindowFitAdvice)
    assert advice.band == "fits"
    assert advice.utilization == pytest.approx(4_000 / 128_000)
    assert advice.headroom_tokens == 124_000
    assert "fits" in advice.advisory.lower()
    assert advice.model == "gpt-5.5"


def test_tight_band_near_context_limit() -> None:
    """Utilization at/above tight_ratio but under limit is tight."""
    advisor = ContextWindowFitAdvisor(tight_ratio=0.85)
    advice = advisor.advise(
        estimated_tokens=110_000,
        model_context_limit=128_000,
        model="claude-sonnet-4.6",
    )
    assert advice.band == "tight"
    assert advice.utilization >= 0.85
    assert advice.headroom_tokens == 18_000
    assert "tight" in advice.advisory.lower()


def test_overflow_when_estimated_meets_or_exceeds_limit() -> None:
    """estimated_tokens >= model_context_limit yields overflow."""
    advisor = ContextWindowFitAdvisor()
    at_limit = advisor.advise(estimated_tokens=200_000, model_context_limit=200_000)
    assert at_limit.band == "overflow"
    assert at_limit.headroom_tokens == 0
    assert "overflow" in at_limit.advisory.lower()

    over = advisor.advise(
        estimated_tokens=250_000,
        model_context_limit=200_000,
        model="gemini-3.1-pro-preview",
    )
    assert over.band == "overflow"
    assert over.headroom_tokens == -50_000
    assert over.model == "gemini-3.1-pro-preview"


def test_frontier_models_share_advisor_thresholds() -> None:
    """Same thresholds apply across GPT-5.5 / Sonnet / Gemini / Kimi."""
    advisor = ContextWindowFitAdvisor(tight_ratio=0.9)
    for model in (
        "gpt-5.5",
        "claude-sonnet-4.6",
        "gemini-3.1-pro-preview",
        "kimi-k2",
    ):
        advice = advisor.advise(
            estimated_tokens=90_000,
            model_context_limit=100_000,
            model=model,
        )
        assert advice.band == "tight"
        assert advice.model == model


def test_rejects_invalid_inputs() -> None:
    """Constructor and advise validate ratios and non-negative token counts."""
    with pytest.raises(ValueError, match="tight_ratio"):
        ContextWindowFitAdvisor(tight_ratio=0.0)
    with pytest.raises(ValueError, match="tight_ratio"):
        ContextWindowFitAdvisor(tight_ratio=1.0)
    advisor = ContextWindowFitAdvisor()
    with pytest.raises(ValueError, match="estimated_tokens"):
        advisor.advise(estimated_tokens=-1, model_context_limit=1000)
    with pytest.raises(ValueError, match="model_context_limit"):
        advisor.advise(estimated_tokens=10, model_context_limit=0)
