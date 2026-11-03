"""Tests for SlidingWindowAttentionBleedBandAdvisor."""

from __future__ import annotations

import pytest

from safety.sliding_window_attention_bleed_band import SlidingWindowAttentionBleedBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = SlidingWindowAttentionBleedBandAdvisor().advise(
        request_id="r1", bleed_ratio=0.1, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = SlidingWindowAttentionBleedBandAdvisor().advise(
        request_id="r1", bleed_ratio=0.3, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = SlidingWindowAttentionBleedBandAdvisor().advise(
        request_id="r1", bleed_ratio=0.6, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        SlidingWindowAttentionBleedBandAdvisor().advise(request_id=" ", bleed_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="bleed_ratio"):
        SlidingWindowAttentionBleedBandAdvisor().advise(request_id="r1", bleed_ratio=-0.1)
