"""Tests for KvCacheOffloadThrashBandAdvisor."""

from __future__ import annotations

import pytest

from safety.kv_cache_offload_thrash_band import KvCacheOffloadThrashBandAdvisor


def test_within() -> None:
    """Within soft."""

    advice = KvCacheOffloadThrashBandAdvisor().advise(request_id="r1", thrash_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Soft band."""

    advice = KvCacheOffloadThrashBandAdvisor().advise(request_id="r1", thrash_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """Breach band."""

    advice = KvCacheOffloadThrashBandAdvisor().advise(request_id="r1", thrash_ratio=0.9)
    assert advice.band == "breach"


def test_invalid_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        KvCacheOffloadThrashBandAdvisor().advise(request_id=" ", thrash_ratio=0.1)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="thrash_ratio"):
        KvCacheOffloadThrashBandAdvisor().advise(request_id="r1", thrash_ratio=-0.1)
