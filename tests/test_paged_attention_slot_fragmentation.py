"""Tests for PagedAttentionSlotFragmentationAdvisor."""

from __future__ import annotations

import pytest

from safety.paged_attention_slot_fragmentation import (
    PagedAttentionSlotFragmentationAdvisor,
)


def test_within() -> None:
    """Band within."""

    advice = PagedAttentionSlotFragmentationAdvisor().advise(
        request_id="r1",
        slot_fragmentation_ratio=0.0,
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.2 + 0.5) / 2
    advice = PagedAttentionSlotFragmentationAdvisor().advise(
        request_id="r1",
        slot_fragmentation_ratio=mid,
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = PagedAttentionSlotFragmentationAdvisor().advise(
        request_id="r1",
        slot_fragmentation_ratio=1.0000,
    )
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError, match="request_id"):
        PagedAttentionSlotFragmentationAdvisor().advise(
            request_id=" ",
            slot_fragmentation_ratio=0.0,
        )
