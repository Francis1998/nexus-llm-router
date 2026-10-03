"""Unit tests for KvCacheEvictionPressureAdvisor."""

from __future__ import annotations

import pytest

from safety.kv_cache_eviction_pressure import KvCacheEvictionPressureAdvisor


def test_within() -> None:
    """Low metric is within."""

    advice = KvCacheEvictionPressureAdvisor().advise(request_id="r1", eviction_rate=0.02)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = KvCacheEvictionPressureAdvisor().advise(request_id="r1", eviction_rate=0.1)
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = KvCacheEvictionPressureAdvisor().advise(request_id="r1", eviction_rate=0.25)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="eviction_rate"):
        KvCacheEvictionPressureAdvisor().advise(request_id="r1", eviction_rate=-0.1)
