"""Unit tests for PrefixCacheEvictionStormAdvisor."""

from __future__ import annotations

import pytest

from safety.prefix_cache_eviction_storm import (
    PrefixCacheEvictionStormAdvisor,
)


def test_within() -> None:
    """Below soft -> within."""

    advice = PrefixCacheEvictionStormAdvisor().advise(request_id="r1", eviction_storm_ratio=0.05)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard -> soft."""

    advice = PrefixCacheEvictionStormAdvisor().advise(request_id="r1", eviction_storm_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard -> breach."""

    advice = PrefixCacheEvictionStormAdvisor().advise(request_id="r1", eviction_storm_ratio=0.8)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        PrefixCacheEvictionStormAdvisor().advise(request_id=" ", eviction_storm_ratio=0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        PrefixCacheEvictionStormAdvisor().advise(
            request_id="r1",
            eviction_storm_ratio=0.1,
            soft_limit=0.5,
            hard_limit=0.2,
        )
