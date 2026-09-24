"""Unit tests for ProviderExplorationEpsilonAdvisor."""

from __future__ import annotations

import pytest

from safety.provider_exploration_epsilon import ProviderExplorationEpsilonAdvisor


def test_exploit_when_sample_above_epsilon() -> None:
    """Sample >= epsilon exploits sticky/best provider."""

    advice = ProviderExplorationEpsilonAdvisor().advise(epsilon=0.1, sample=0.5)
    assert advice.band == "exploit"
    assert advice.explore is False


def test_explore_when_sample_below_epsilon() -> None:
    """Sample < epsilon explores."""

    advice = ProviderExplorationEpsilonAdvisor().advise(epsilon=0.2, sample=0.1)
    assert advice.band == "explore"
    assert advice.explore is True


def test_forced_explore_at_epsilon_one() -> None:
    """Epsilon 1.0 forces explore."""

    advice = ProviderExplorationEpsilonAdvisor().advise(epsilon=1.0, sample=0.99)
    assert advice.band == "forced_explore"


def test_invalid_epsilon_raises() -> None:
    """Epsilon outside [0,1] raises ValueError."""

    with pytest.raises(ValueError, match="epsilon"):
        ProviderExplorationEpsilonAdvisor().advise(epsilon=1.5, sample=0.1)
