"""Tests for RequestCostForecastAdvisor offline USD estimates."""

from __future__ import annotations

import pytest

from safety.cost_forecast import CostForecast, RequestCostForecastAdvisor


def test_forecast_splits_prompt_and_completion_cost() -> None:
    """USD estimate uses (tokens/1e6) * $/1M for prompt and completion."""
    advisor = RequestCostForecastAdvisor()
    forecast = advisor.forecast(
        prompt_tokens=1_000,
        max_completion_tokens=500,
        input_usd_per_1m=5.0,
        output_usd_per_1m=15.0,
        model="gpt-5.5",
    )
    assert isinstance(forecast, CostForecast)
    assert forecast.prompt_cost == pytest.approx(0.005)
    assert forecast.completion_cost == pytest.approx(0.0075)
    assert forecast.estimated_usd == pytest.approx(0.0125)
    assert forecast.assumptions["assumes_full_max_completion"] is True
    assert forecast.assumptions["model"] == "gpt-5.5"
    assert forecast.assumptions["input_usd_per_1m"] == 5.0


def test_default_rates_apply_across_frontier_models() -> None:
    """Advisor defaults cover GPT-5.5 / Sonnet / Gemini / Kimi forecasts."""
    advisor = RequestCostForecastAdvisor(
        default_input_usd_per_1m=3.0,
        default_output_usd_per_1m=15.0,
    )
    for model in (
        "gpt-5.5",
        "claude-sonnet-4.6",
        "gemini-3.1-pro-preview",
        "kimi-k2",
    ):
        forecast = advisor.forecast(
            prompt_tokens=2_000,
            max_completion_tokens=1_000,
            model=model,
        )
        assert forecast.prompt_cost == pytest.approx(0.006)
        assert forecast.completion_cost == pytest.approx(0.015)
        assert forecast.estimated_usd == pytest.approx(0.021)
        assert forecast.assumptions["model"] == model


def test_explicit_rates_override_defaults() -> None:
    """Per-call rates win over constructor defaults."""
    advisor = RequestCostForecastAdvisor(
        default_input_usd_per_1m=1.0,
        default_output_usd_per_1m=1.0,
    )
    forecast = advisor.forecast(
        prompt_tokens=1_000_000,
        max_completion_tokens=1_000_000,
        input_usd_per_1m=10.0,
        output_usd_per_1m=20.0,
    )
    assert forecast.prompt_cost == pytest.approx(10.0)
    assert forecast.completion_cost == pytest.approx(20.0)
    assert forecast.estimated_usd == pytest.approx(30.0)


def test_zero_tokens_yields_zero_cost() -> None:
    """Zero token counts produce a zero estimate without error."""
    advisor = RequestCostForecastAdvisor()
    forecast = advisor.forecast(
        prompt_tokens=0,
        max_completion_tokens=0,
        input_usd_per_1m=5.0,
        output_usd_per_1m=15.0,
    )
    assert forecast.estimated_usd == 0.0
    assert forecast.prompt_cost == 0.0
    assert forecast.completion_cost == 0.0


def test_rejects_invalid_inputs() -> None:
    """Missing rates and negative inputs raise ValueError."""
    with pytest.raises(ValueError, match="default_input"):
        RequestCostForecastAdvisor(default_input_usd_per_1m=-1.0)
    advisor = RequestCostForecastAdvisor()
    with pytest.raises(ValueError, match="prompt_tokens"):
        advisor.forecast(
            prompt_tokens=-1,
            max_completion_tokens=10,
            input_usd_per_1m=1.0,
            output_usd_per_1m=1.0,
        )
    with pytest.raises(ValueError, match="max_completion_tokens"):
        advisor.forecast(
            prompt_tokens=10,
            max_completion_tokens=-1,
            input_usd_per_1m=1.0,
            output_usd_per_1m=1.0,
        )
    with pytest.raises(ValueError, match="input_usd_per_1m is required"):
        advisor.forecast(prompt_tokens=10, max_completion_tokens=10)
    with pytest.raises(ValueError, match="input_usd_per_1m must be >= 0"):
        advisor.forecast(
            prompt_tokens=10,
            max_completion_tokens=10,
            input_usd_per_1m=-0.1,
            output_usd_per_1m=1.0,
        )
    with pytest.raises(ValueError, match="output_usd_per_1m must be >= 0"):
        advisor.forecast(
            prompt_tokens=10,
            max_completion_tokens=10,
            input_usd_per_1m=1.0,
            output_usd_per_1m=-0.1,
        )
