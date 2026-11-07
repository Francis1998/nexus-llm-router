"""Tests for LoraRankSkewBandAdvisor."""

from __future__ import annotations

import pytest

from safety.lora_rank_skew_band import LoraRankSkewBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = LoraRankSkewBandAdvisor().advise(
        request_id="req-1", skew_ratio=0.1, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = LoraRankSkewBandAdvisor().advise(
        request_id="req-1", skew_ratio=0.3, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = LoraRankSkewBandAdvisor().advise(
        request_id="req-1", skew_ratio=0.7, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        LoraRankSkewBandAdvisor().advise(request_id=" ", skew_ratio=0.1)


def test_negative_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="skew_ratio"):
        LoraRankSkewBandAdvisor().advise(request_id="req-1", skew_ratio=-0.1)
