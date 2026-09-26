"""Unit tests for CostSpikeAnomalyAdvisor."""

from __future__ import annotations

import pytest

from safety.cost_spike_anomaly import CostSpikeAnomalyAdvisor


def test_normal_band() -> None:
    """Spend near baseline is normal."""

    advice = CostSpikeAnomalyAdvisor().advise(
        tenant_id="t1",
        current_spend_usd=10.0,
        baseline_spend_usd=10.0,
    )
    assert advice.band == "normal"


def test_elevated_band() -> None:
    """1.5x+ spend is elevated."""

    advice = CostSpikeAnomalyAdvisor().advise(
        tenant_id="t1",
        current_spend_usd=20.0,
        baseline_spend_usd=10.0,
    )
    assert advice.band == "elevated"


def test_anomalous_band() -> None:
    """3x+ spend is anomalous."""

    advice = CostSpikeAnomalyAdvisor().advise(
        tenant_id="t1",
        current_spend_usd=40.0,
        baseline_spend_usd=10.0,
    )
    assert advice.band == "anomalous"


def test_empty_tenant_raises() -> None:
    """Empty tenant id raises ValueError."""

    with pytest.raises(ValueError, match="tenant_id"):
        CostSpikeAnomalyAdvisor().advise(
            tenant_id=" ",
            current_spend_usd=1.0,
            baseline_spend_usd=1.0,
        )
