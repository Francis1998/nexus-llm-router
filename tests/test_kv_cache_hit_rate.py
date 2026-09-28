"""Unit tests for KvCacheHitRateAdvisor."""

from __future__ import annotations

import pytest

from safety.kv_cache_hit_rate import KvCacheHitRateAdvisor


def test_healthy_band() -> None:
    """At/over target is healthy."""

    advice = KvCacheHitRateAdvisor().advise(request_id="r1", hit_rate=0.9, target_rate=0.8)
    assert advice.band == "healthy"


def test_soft_band() -> None:
    """70%+ of target is soft."""

    advice = KvCacheHitRateAdvisor().advise(request_id="r1", hit_rate=0.6, target_rate=0.8)
    assert advice.band == "soft"


def test_cold_band() -> None:
    """Under 70% of target is cold."""

    advice = KvCacheHitRateAdvisor().advise(request_id="r1", hit_rate=0.2, target_rate=0.8)
    assert advice.band == "cold"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        KvCacheHitRateAdvisor().advise(request_id=" ", hit_rate=0.5, target_rate=0.5)
