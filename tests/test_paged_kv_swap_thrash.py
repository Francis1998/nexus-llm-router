"""Tests for PagedKvSwapThrashAdvisor."""

from __future__ import annotations

import pytest

from safety.paged_kv_swap_thrash import PagedKvSwapThrashAdvisor


def test_within_band() -> None:
    """Below soft is within."""

    advice = PagedKvSwapThrashAdvisor().advise(request_id="r1", swap_thrash_ratio=0.0)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Between soft and hard is soft."""

    advice = PagedKvSwapThrashAdvisor().advise(request_id="r1", swap_thrash_ratio=0.35)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """At/above hard is breach."""

    advice = PagedKvSwapThrashAdvisor().advise(request_id="r1", swap_thrash_ratio=0.5)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Blank request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        PagedKvSwapThrashAdvisor().advise(request_id=" ", swap_thrash_ratio=0.0)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="swap_thrash_ratio"):
        PagedKvSwapThrashAdvisor().advise(request_id="r1", swap_thrash_ratio=-0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        PagedKvSwapThrashAdvisor().advise(
            request_id="r1",
            swap_thrash_ratio=0.0,
            soft_limit=10.0,
            hard_limit=1.0,
        )
