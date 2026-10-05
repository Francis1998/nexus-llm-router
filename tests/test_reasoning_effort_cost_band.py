"""Unit tests for ReasoningEffortCostBandAdvisor."""

from __future__ import annotations

import pytest

from safety.reasoning_effort_cost_band import ReasoningEffortCostBandAdvisor


def test_within() -> None:
    """Low metric is within."""

    advice = ReasoningEffortCostBandAdvisor().advise(
        request_id="r1", effort_cost_usd=0.020000000000000004
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = ReasoningEffortCostBandAdvisor().advise(request_id="r1", effort_cost_usd=0.15)
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = ReasoningEffortCostBandAdvisor().advise(request_id="r1", effort_cost_usd=0.3)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="effort_cost_usd"):
        ReasoningEffortCostBandAdvisor().advise(request_id="r1", effort_cost_usd=-0.1)
