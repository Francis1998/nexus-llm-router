"""Unit tests for ProviderColdStartLatencyAdvisor."""

from __future__ import annotations

import pytest

from safety.provider_cold_start_latency import ProviderColdStartLatencyAdvisor


def test_warm_band() -> None:
    """Low latency is warm."""

    advice = ProviderColdStartLatencyAdvisor().advise(
        provider="openai",
        cold_start_ms=100.0,
    )
    assert advice.band == "warm"


def test_cooling_band() -> None:
    """Mid latency is cooling."""

    advice = ProviderColdStartLatencyAdvisor().advise(
        provider="openai",
        cold_start_ms=500.0,
    )
    assert advice.band == "cooling"


def test_cold_band() -> None:
    """High latency is cold."""

    advice = ProviderColdStartLatencyAdvisor().advise(
        provider="openai",
        cold_start_ms=2000.0,
    )
    assert advice.band == "cold"


def test_empty_provider_raises() -> None:
    """Empty provider raises ValueError."""

    with pytest.raises(ValueError, match="provider"):
        ProviderColdStartLatencyAdvisor().advise(provider="  ", cold_start_ms=10.0)
