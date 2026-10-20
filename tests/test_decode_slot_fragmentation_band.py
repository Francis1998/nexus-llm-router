"""Tests for DecodeSlotFragmentationBandAdvisor."""

from __future__ import annotations

import pytest

from safety.decode_slot_fragmentation_band import DecodeSlotFragmentationBandAdvisor


def test_within_band() -> None:
    """Below soft is within."""

    advice = DecodeSlotFragmentationBandAdvisor().advise(request_id="r1", slot_frag_ratio=0.0)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Between soft and hard is soft."""

    advice = DecodeSlotFragmentationBandAdvisor().advise(request_id="r1", slot_frag_ratio=0.465)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """At/above hard is breach."""

    advice = DecodeSlotFragmentationBandAdvisor().advise(request_id="r1", slot_frag_ratio=0.65)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Blank request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        DecodeSlotFragmentationBandAdvisor().advise(request_id=" ", slot_frag_ratio=0.0)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="slot_frag_ratio"):
        DecodeSlotFragmentationBandAdvisor().advise(request_id="r1", slot_frag_ratio=-0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        DecodeSlotFragmentationBandAdvisor().advise(
            request_id="r1",
            slot_frag_ratio=0.0,
            soft_limit=10.0,
            hard_limit=1.0,
        )
