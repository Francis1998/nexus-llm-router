"""Unit tests for StickySessionBleedAdvisor."""

from __future__ import annotations

import pytest

from safety.sticky_session_bleed import StickySessionBleedAdvisor


def test_within_band() -> None:
    """Low foreign hit rate is within."""

    advice = StickySessionBleedAdvisor().advise(request_id="r1", foreign_hit_rate=0.01)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Mid foreign hit rate is soft."""

    advice = StickySessionBleedAdvisor().advise(request_id="r1", foreign_hit_rate=0.1)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """High foreign hit rate is breach."""

    advice = StickySessionBleedAdvisor().advise(request_id="r1", foreign_hit_rate=0.5)
    assert advice.band == "breach"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        StickySessionBleedAdvisor().advise(request_id=" ", foreign_hit_rate=0.0)
