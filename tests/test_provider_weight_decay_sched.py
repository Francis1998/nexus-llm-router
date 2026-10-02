"""Unit tests for ProviderWeightDecaySchedulerAdvisor."""

from __future__ import annotations

import pytest

from safety.provider_weight_decay_sched import ProviderWeightDecaySchedulerAdvisor


def test_within() -> None:
    """Small decay is within."""

    advice = ProviderWeightDecaySchedulerAdvisor().advise(request_id="r1", decay_step=0.01)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid decay is soft."""

    advice = ProviderWeightDecaySchedulerAdvisor().advise(request_id="r1", decay_step=0.1)
    assert advice.band == "soft"


def test_breach() -> None:
    """Large decay is breach."""

    advice = ProviderWeightDecaySchedulerAdvisor().advise(request_id="r1", decay_step=0.5)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative decay raises."""

    with pytest.raises(ValueError, match="decay_step"):
        ProviderWeightDecaySchedulerAdvisor().advise(request_id="r1", decay_step=-0.1)
