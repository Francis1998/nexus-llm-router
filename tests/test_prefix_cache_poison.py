"""Tests for PrefixCachePoisonAdvisor."""

from __future__ import annotations

import pytest

from safety.prefix_cache_poison import PrefixCachePoisonAdvisor


def test_within() -> None:
    """Band within."""

    advice = PrefixCachePoisonAdvisor().advise(request_id="r1", poison_hit_rate=0.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.2 + 0.5) / 2
    advice = PrefixCachePoisonAdvisor().advise(request_id="r1", poison_hit_rate=mid)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = PrefixCachePoisonAdvisor().advise(request_id="r1", poison_hit_rate=1.0000)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError, match="request_id"):
        PrefixCachePoisonAdvisor().advise(request_id=" ", poison_hit_rate=0.0)
