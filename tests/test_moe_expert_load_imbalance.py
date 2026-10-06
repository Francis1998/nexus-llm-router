"""Unit tests for MoeExpertLoadImbalanceAdvisor."""

from __future__ import annotations

import pytest

from safety.moe_expert_load_imbalance import MoeExpertLoadImbalanceAdvisor


def test_within() -> None:
    """Low metric is within."""

    advice = MoeExpertLoadImbalanceAdvisor().advise(request_id="r1", imbalance_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = MoeExpertLoadImbalanceAdvisor().advise(request_id="r1", imbalance_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = MoeExpertLoadImbalanceAdvisor().advise(request_id="r1", imbalance_ratio=0.6)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="imbalance_ratio"):
        MoeExpertLoadImbalanceAdvisor().advise(request_id="r1", imbalance_ratio=-0.1)
