"""Unit tests for ProviderCircuitBreakerCooldownAdvisor."""

from __future__ import annotations

import pytest

from safety.provider_circuit_breaker_cooldown import ProviderCircuitBreakerCooldownAdvisor


def test_within() -> None:
    """Band within."""

    advice = ProviderCircuitBreakerCooldownAdvisor().advise(request_id="r1", cooldown_ratio=0.15)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = ProviderCircuitBreakerCooldownAdvisor().advise(request_id="r1", cooldown_ratio=0.5)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = ProviderCircuitBreakerCooldownAdvisor().advise(request_id="r1", cooldown_ratio=1.7)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="cooldown_ratio"):
        ProviderCircuitBreakerCooldownAdvisor().advise(request_id="r1", cooldown_ratio=-0.1)
