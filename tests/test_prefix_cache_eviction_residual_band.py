"""Tests for PrefixCacheEvictionResidualBandAdvisor."""

from __future__ import annotations

import pytest

from safety.prefix_cache_eviction_residual_band import PrefixCacheEvictionResidualBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = PrefixCacheEvictionResidualBandAdvisor().advise(
        request_id="r1", eviction_residual_ratio=0.1
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = PrefixCacheEvictionResidualBandAdvisor().advise(
        request_id="r1", eviction_residual_ratio=0.3
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = PrefixCacheEvictionResidualBandAdvisor().advise(
        request_id="r1", eviction_residual_ratio=0.9
    )
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        PrefixCacheEvictionResidualBandAdvisor().advise(request_id=" ", eviction_residual_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="eviction_residual_ratio"):
        PrefixCacheEvictionResidualBandAdvisor().advise(
            request_id="r1", eviction_residual_ratio=-0.1
        )
