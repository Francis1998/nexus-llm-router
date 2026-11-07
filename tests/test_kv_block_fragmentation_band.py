"""Tests for KvBlockFragmentationBandAdvisor."""

from __future__ import annotations

import pytest

from safety.kv_block_fragmentation_band import KvBlockFragmentationBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = KvBlockFragmentationBandAdvisor().advise(
        request_id="req-1", fragmentation_ratio=0.1, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = KvBlockFragmentationBandAdvisor().advise(
        request_id="req-1", fragmentation_ratio=0.3, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = KvBlockFragmentationBandAdvisor().advise(
        request_id="req-1", fragmentation_ratio=0.7, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        KvBlockFragmentationBandAdvisor().advise(request_id=" ", fragmentation_ratio=0.1)


def test_negative_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="fragmentation_ratio"):
        KvBlockFragmentationBandAdvisor().advise(request_id="req-1", fragmentation_ratio=-0.1)
