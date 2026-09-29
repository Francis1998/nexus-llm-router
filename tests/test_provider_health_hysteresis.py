"""Unit tests for ProviderHealthHysteresisAdvisor."""

from __future__ import annotations

import pytest

from safety.provider_health_hysteresis import ProviderHealthHysteresisAdvisor


def test_healthy_when_low() -> None:
    """Low error rate while healthy stays healthy."""

    advice = ProviderHealthHysteresisAdvisor().advise(
        provider_id="p1",
        error_rate=0.01,
        enter_threshold=0.1,
        exit_threshold=0.05,
        currently_unhealthy=False,
    )
    assert advice.band == "healthy"


def test_enter_unhealthy() -> None:
    """Crossing enter threshold marks unhealthy."""

    advice = ProviderHealthHysteresisAdvisor().advise(
        provider_id="p1",
        error_rate=0.2,
        enter_threshold=0.1,
        exit_threshold=0.05,
        currently_unhealthy=False,
    )
    assert advice.band == "unhealthy"


def test_hold_while_recovering() -> None:
    """Between exit and enter while unhealthy holds."""

    advice = ProviderHealthHysteresisAdvisor().advise(
        provider_id="p1",
        error_rate=0.08,
        enter_threshold=0.1,
        exit_threshold=0.05,
        currently_unhealthy=True,
    )
    assert advice.band == "hold"


def test_empty_provider_raises() -> None:
    """Empty provider id raises ValueError."""

    with pytest.raises(ValueError, match="provider_id"):
        ProviderHealthHysteresisAdvisor().advise(
            provider_id=" ",
            error_rate=0.01,
            enter_threshold=0.1,
            exit_threshold=0.05,
            currently_unhealthy=False,
        )
