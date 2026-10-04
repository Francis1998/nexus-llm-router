"""Unit tests for ToolChoiceStickyAffinityAdvisor."""

from __future__ import annotations

import pytest

from safety.tool_choice_sticky_affinity import ToolChoiceStickyAffinityAdvisor


def test_within() -> None:
    """Low metric is within."""

    advice = ToolChoiceStickyAffinityAdvisor().advise(request_id="r1", affinity_drift=0.05)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = ToolChoiceStickyAffinityAdvisor().advise(request_id="r1", affinity_drift=0.2)
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = ToolChoiceStickyAffinityAdvisor().advise(request_id="r1", affinity_drift=0.5)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="affinity_drift"):
        ToolChoiceStickyAffinityAdvisor().advise(request_id="r1", affinity_drift=-0.1)
