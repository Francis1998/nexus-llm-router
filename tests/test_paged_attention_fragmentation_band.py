"""Unit tests for PagedAttentionFragmentationBandAdvisor."""

from __future__ import annotations

import pytest

from safety.paged_attention_fragmentation_band import (
    PagedAttentionFragmentationBandAdvisor,
)


def test_within() -> None:
    """Below soft -> within."""

    advice = PagedAttentionFragmentationBandAdvisor().advise(
        request_id="r1", fragmentation_ratio=0.1250
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard -> soft."""

    advice = PagedAttentionFragmentationBandAdvisor().advise(
        request_id="r1", fragmentation_ratio=0.4000
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard -> breach."""

    advice = PagedAttentionFragmentationBandAdvisor().advise(
        request_id="r1", fragmentation_ratio=0.7500
    )
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        PagedAttentionFragmentationBandAdvisor().advise(request_id=" ", fragmentation_ratio=0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        PagedAttentionFragmentationBandAdvisor().advise(
            request_id="r1",
            fragmentation_ratio=0.1,
            soft_limit=0.5,
            hard_limit=0.2,
        )
