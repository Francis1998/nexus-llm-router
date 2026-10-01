"""Unit tests for LoraAdapterSwapCostAdvisor."""

from __future__ import annotations

import pytest

from safety.lora_adapter_swap_cost import LoraAdapterSwapCostAdvisor


def test_within_band() -> None:
    """Low swap cost is within."""

    advice = LoraAdapterSwapCostAdvisor().advise(request_id="r1", swap_ms=10.0)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Mid swap cost is soft."""

    advice = LoraAdapterSwapCostAdvisor().advise(request_id="r1", swap_ms=80.0)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """High swap cost is breach."""

    advice = LoraAdapterSwapCostAdvisor().advise(request_id="r1", swap_ms=250.0)
    assert advice.band == "breach"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        LoraAdapterSwapCostAdvisor().advise(request_id=" ", swap_ms=1.0)
