"""Tests for LoraAdapterHotSwapStallAdvisor."""

from __future__ import annotations

import pytest

from safety.lora_adapter_hot_swap_stall import LoraAdapterHotSwapStallAdvisor


def test_within() -> None:
    """Band within."""

    advice = LoraAdapterHotSwapStallAdvisor().advise(request_id="r1", stall_ratio=0.1000)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.2 + 0.5) / 2
    advice = LoraAdapterHotSwapStallAdvisor().advise(request_id="r1", stall_ratio=mid)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = LoraAdapterHotSwapStallAdvisor().advise(request_id="r1", stall_ratio=1.0000)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError):
        LoraAdapterHotSwapStallAdvisor().advise(request_id="r1", stall_ratio=-0.1)
