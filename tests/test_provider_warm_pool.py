"""Tests for ProviderWarmPoolAdvisor."""

from __future__ import annotations

import pytest

from safety.provider_warm_pool import ProviderWarmPoolAdvisor


def test_hot() -> None:
    advice = ProviderWarmPoolAdvisor().advise("p1", warm_replicas=2, target_replicas=2)
    assert advice.band == "hot"


def test_warming() -> None:
    advice = ProviderWarmPoolAdvisor().advise("p1", warm_replicas=1, target_replicas=2)
    assert advice.band == "warming"


def test_cold() -> None:
    advice = ProviderWarmPoolAdvisor().advise("p1", warm_replicas=0, target_replicas=4)
    assert advice.band == "cold"


def test_invalid() -> None:
    with pytest.raises(ValueError, match="provider_id"):
        ProviderWarmPoolAdvisor().advise("", warm_replicas=1, target_replicas=1)
    with pytest.raises(ValueError, match="target_replicas"):
        ProviderWarmPoolAdvisor().advise("p", warm_replicas=0, target_replicas=0)
