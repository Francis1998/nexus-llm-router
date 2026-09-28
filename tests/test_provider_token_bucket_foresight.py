"""Unit tests for ProviderTokenBucketForesightAdvisor."""

from __future__ import annotations

import pytest

from safety.provider_token_bucket_foresight import ProviderTokenBucketForesightAdvisor


def test_ample_band() -> None:
    """1.2x+ remaining is ample."""

    advice = ProviderTokenBucketForesightAdvisor().advise(
        provider="openai", tokens_remaining=2400.0, forecast_demand=1000.0
    )
    assert advice.band == "ample"


def test_tight_band() -> None:
    """1.0-1.2x remaining is tight."""

    advice = ProviderTokenBucketForesightAdvisor().advise(
        provider="openai", tokens_remaining=1100.0, forecast_demand=1000.0
    )
    assert advice.band == "tight"


def test_exhausted_band() -> None:
    """Under demand is exhausted."""

    advice = ProviderTokenBucketForesightAdvisor().advise(
        provider="openai", tokens_remaining=500.0, forecast_demand=1000.0
    )
    assert advice.band == "exhausted"


def test_empty_provider_raises() -> None:
    """Empty provider raises ValueError."""

    with pytest.raises(ValueError, match="provider"):
        ProviderTokenBucketForesightAdvisor().advise(
            provider=" ", tokens_remaining=1.0, forecast_demand=1.0
        )
