"""Unit tests for OutputTokenForecastAdvisor."""

from __future__ import annotations

import pytest

from safety.output_token_forecast import OutputTokenForecastAdvisor


def test_fit_band() -> None:
    """Well under budget is fit."""

    advice = OutputTokenForecastAdvisor().advise(
        request_id="r1", estimated_tokens=100, budget_tokens=500
    )
    assert advice.band == "fit"


def test_tight_band() -> None:
    """Near budget is tight."""

    advice = OutputTokenForecastAdvisor().advise(
        request_id="r1", estimated_tokens=420, budget_tokens=500
    )
    assert advice.band == "tight"


def test_over_band() -> None:
    """At/over budget is over."""

    advice = OutputTokenForecastAdvisor().advise(
        request_id="r1", estimated_tokens=500, budget_tokens=500
    )
    assert advice.band == "over"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        OutputTokenForecastAdvisor().advise(request_id=" ", estimated_tokens=1, budget_tokens=1)
