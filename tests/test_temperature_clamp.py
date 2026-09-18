"""Tests for ModelTemperatureClampAdvisor ok/high/extreme advisory bands."""

from __future__ import annotations

import pytest

from safety.temperature_clamp import ModelTemperatureClampAdvisor, TemperatureClampAdvice


def test_ok_band_when_well_under_max_allowed() -> None:
    """Low utilization yields ok with suggested_clamp equal to temperature."""
    advisor = ModelTemperatureClampAdvisor(high_ratio=0.85)
    advice = advisor.advise(temperature=0.4, max_allowed=2.0, model="gpt-5.5")
    assert isinstance(advice, TemperatureClampAdvice)
    assert advice.band == "ok"
    assert advice.suggested_clamp == pytest.approx(0.4)
    assert "ok" in advice.advisory.lower()
    assert advice.model == "gpt-5.5"


def test_high_band_at_high_ratio() -> None:
    """Utilization at/above high_ratio but under max_allowed is high."""
    advisor = ModelTemperatureClampAdvisor(high_ratio=0.85)
    advice = advisor.advise(
        temperature=1.7,
        max_allowed=2.0,
        model="claude-sonnet-4.6",
    )
    assert advice.band == "high"
    assert advice.temperature / advice.max_allowed >= 0.85
    assert advice.suggested_clamp == pytest.approx(1.7)
    assert "high" in advice.advisory.lower()


def test_extreme_band_when_temperature_exceeds_max_allowed() -> None:
    """temperature > max_allowed yields extreme with clamp suggestion."""
    advisor = ModelTemperatureClampAdvisor()
    advice = advisor.advise(
        temperature=2.5,
        max_allowed=2.0,
        model="gemini-3.1-pro-preview",
    )
    assert advice.band == "extreme"
    assert advice.suggested_clamp == pytest.approx(2.0)
    assert "extreme" in advice.advisory.lower()
    assert advice.model == "gemini-3.1-pro-preview"


def test_frontier_models_share_temperature_thresholds() -> None:
    """Same thresholds apply across GPT-5.5 / Sonnet / Gemini / Kimi."""
    advisor = ModelTemperatureClampAdvisor(high_ratio=0.9)
    for model in (
        "gpt-5.5",
        "claude-sonnet-4.6",
        "gemini-3.1-pro-preview",
        "kimi-k2",
    ):
        advice = advisor.advise(temperature=1.8, max_allowed=2.0, model=model)
        assert advice.band == "high"
        assert advice.model == model


def test_rejects_invalid_inputs() -> None:
    """Constructor and advise validate ratios and non-negative temperature."""
    with pytest.raises(ValueError, match="high_ratio"):
        ModelTemperatureClampAdvisor(high_ratio=0.0)
    with pytest.raises(ValueError, match="high_ratio"):
        ModelTemperatureClampAdvisor(high_ratio=1.0)
    advisor = ModelTemperatureClampAdvisor()
    with pytest.raises(ValueError, match="temperature"):
        advisor.advise(temperature=-0.1, max_allowed=2.0)
    with pytest.raises(ValueError, match="max_allowed"):
        advisor.advise(temperature=0.5, max_allowed=0.0)
    # Exact max_allowed is still high (not extreme) — extreme is strictly greater.
    at_max = advisor.advise(temperature=2.0, max_allowed=2.0)
    assert at_max.band == "high"
    assert at_max.suggested_clamp == pytest.approx(2.0)
