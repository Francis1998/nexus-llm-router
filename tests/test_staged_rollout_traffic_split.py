"""Unit tests for StagedRolloutTrafficSplitGuard."""

from __future__ import annotations

import pytest

from safety.staged_rollout_traffic_split import StagedRolloutTrafficSplitGuard


def test_within() -> None:
    """Low metric is within."""

    advice = StagedRolloutTrafficSplitGuard().check(request_id="r1", canary_share=0.05)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = StagedRolloutTrafficSplitGuard().check(request_id="r1", canary_share=0.2)
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = StagedRolloutTrafficSplitGuard().check(request_id="r1", canary_share=0.4)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="canary_share"):
        StagedRolloutTrafficSplitGuard().check(request_id="r1", canary_share=-0.1)
