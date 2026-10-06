"""Unit tests for ContextWindowFragmentationAdvisor."""

from __future__ import annotations

import pytest

from safety.context_window_fragmentation import ContextWindowFragmentationAdvisor


def test_within() -> None:
    """Low metric is within."""

    advice = ContextWindowFragmentationAdvisor().advise(request_id="r1", fragmentation_ratio=0.05)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = ContextWindowFragmentationAdvisor().advise(request_id="r1", fragmentation_ratio=0.25)
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = ContextWindowFragmentationAdvisor().advise(request_id="r1", fragmentation_ratio=0.5)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="fragmentation_ratio"):
        ContextWindowFragmentationAdvisor().advise(request_id="r1", fragmentation_ratio=-0.1)
