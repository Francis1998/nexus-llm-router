"""Unit tests for PrefillTtftRatioAdvisor."""

from __future__ import annotations

import pytest

from safety.prefill_ttft_ratio import PrefillTtftRatioAdvisor


def test_balanced_band() -> None:
    """Low prefill share is balanced."""

    advice = PrefillTtftRatioAdvisor().advise(
        provider="openai",
        prefill_ms=100.0,
        ttft_ms=400.0,
    )
    assert advice.band == "balanced"


def test_prefill_heavy_band() -> None:
    """Mid prefill share is prefill_heavy."""

    advice = PrefillTtftRatioAdvisor().advise(
        provider="openai",
        prefill_ms=300.0,
        ttft_ms=400.0,
    )
    assert advice.band == "prefill_heavy"


def test_pathological_band() -> None:
    """Very high prefill share is pathological."""

    advice = PrefillTtftRatioAdvisor().advise(
        provider="openai",
        prefill_ms=380.0,
        ttft_ms=400.0,
    )
    assert advice.band == "pathological"


def test_empty_provider_raises() -> None:
    """Empty provider raises ValueError."""

    with pytest.raises(ValueError, match="provider"):
        PrefillTtftRatioAdvisor().advise(provider=" ", prefill_ms=1.0, ttft_ms=2.0)
