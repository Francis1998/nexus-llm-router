"""Tests for PrefixCacheHitFloorAdvisor."""

from __future__ import annotations

import pytest

from safety.prefix_cache_hit_floor import PrefixCacheHitFloorAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = PrefixCacheHitFloorAdvisor().advise(request_id="r1", miss_ratio=0.01)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = PrefixCacheHitFloorAdvisor().advise(
        request_id="r1",
        miss_ratio=0.2,
        soft_limit=0.2,
        hard_limit=0.55,
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = PrefixCacheHitFloorAdvisor().advise(
        request_id="r1",
        miss_ratio=0.55,
        soft_limit=0.2,
        hard_limit=0.55,
    )
    assert advice.band == "breach"


def test_invalid() -> None:
    """Invalid inputs raise."""

    with pytest.raises(ValueError, match="request_id"):
        PrefixCacheHitFloorAdvisor().advise(request_id=" ", miss_ratio=0.1)
