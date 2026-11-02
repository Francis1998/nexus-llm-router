"""Tests for DisaggScheduleBubbleBandAdvisor."""

from __future__ import annotations

import pytest

from safety.disagg_schedule_bubble_band import DisaggScheduleBubbleBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = DisaggScheduleBubbleBandAdvisor().advise(request_id="r1", bubble_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = DisaggScheduleBubbleBandAdvisor().advise(request_id="r1", bubble_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = DisaggScheduleBubbleBandAdvisor().advise(request_id="r1", bubble_ratio=0.9)
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        DisaggScheduleBubbleBandAdvisor().advise(request_id=" ", bubble_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="bubble_ratio"):
        DisaggScheduleBubbleBandAdvisor().advise(request_id="r1", bubble_ratio=-0.1)
