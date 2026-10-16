"""Tests for KvCacheQuantErrorBandAdvisor."""

from __future__ import annotations

import pytest

from safety.kv_cache_quant_error_band import KvCacheQuantErrorBandAdvisor


def test_within() -> None:
    """Band within."""

    advice = KvCacheQuantErrorBandAdvisor().advise(request_id="r1", error_rate=0.1000)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.2 + 0.5) / 2
    advice = KvCacheQuantErrorBandAdvisor().advise(request_id="r1", error_rate=mid)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = KvCacheQuantErrorBandAdvisor().advise(request_id="r1", error_rate=1.0000)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError):
        KvCacheQuantErrorBandAdvisor().advise(request_id="r1", error_rate=-0.1)
