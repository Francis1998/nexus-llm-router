"""Tests for PagedKvEvictionSkewBandAdvisor."""

from __future__ import annotations

import pytest

from safety.paged_kv_eviction_skew_band import PagedKvEvictionSkewBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = PagedKvEvictionSkewBandAdvisor().advise(
        request_id="r1", skew_ratio=0.1, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = PagedKvEvictionSkewBandAdvisor().advise(
        request_id="r1", skew_ratio=0.3, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = PagedKvEvictionSkewBandAdvisor().advise(
        request_id="r1", skew_ratio=0.9, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        PagedKvEvictionSkewBandAdvisor().advise(request_id=" ", skew_ratio=0.1)


def test_bad_limits() -> None:
    """Invalid limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        PagedKvEvictionSkewBandAdvisor().advise(
            request_id="r1", skew_ratio=0.1, soft_limit=0.5, hard_limit=0.2
        )
