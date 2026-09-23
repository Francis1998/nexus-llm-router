"""Tests for AdaptiveRetryJitterAdvisor."""

from __future__ import annotations

import pytest

from safety.adaptive_retry_jitter import AdaptiveRetryJitterAdvisor


def test_low() -> None:
    advice = AdaptiveRetryJitterAdvisor().advise(attempt=0, base_delay_ms=100.0)
    assert advice.band == "low"
    assert advice.jitter_ms == 25.0


def test_medium() -> None:
    advice = AdaptiveRetryJitterAdvisor().advise(attempt=2, base_delay_ms=100.0)
    assert advice.band == "medium"


def test_high() -> None:
    advice = AdaptiveRetryJitterAdvisor().advise(attempt=4, base_delay_ms=100.0)
    assert advice.band == "high"


def test_invalid() -> None:
    with pytest.raises(ValueError, match="attempt"):
        AdaptiveRetryJitterAdvisor().advise(attempt=-1)
