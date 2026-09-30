"""Unit tests for PrefixCacheThrashAdvisor."""

from __future__ import annotations

import pytest

from safety.prefix_cache_thrash import PrefixCacheThrashAdvisor


def test_within_band() -> None:
    """Low thrash is within."""

    advice = PrefixCacheThrashAdvisor().advise(request_id="r1", evict_count=1, hit_count=10)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Mid thrash is soft."""

    advice = PrefixCacheThrashAdvisor().advise(request_id="r1", evict_count=6, hit_count=10)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """High thrash is breach."""

    advice = PrefixCacheThrashAdvisor().advise(request_id="r1", evict_count=30, hit_count=10)
    assert advice.band == "breach"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        PrefixCacheThrashAdvisor().advise(request_id=" ", evict_count=0, hit_count=1)
