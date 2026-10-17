"""Tests for AttentionSinkDriftAdvisor."""

from __future__ import annotations

import pytest

from safety.attention_sink_drift import AttentionSinkDriftAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = AttentionSinkDriftAdvisor().advise(request_id="r1", sink_drift_ratio=0.01)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = AttentionSinkDriftAdvisor().advise(
        request_id="r1",
        sink_drift_ratio=0.2,
        soft_limit=0.2,
        hard_limit=0.5,
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = AttentionSinkDriftAdvisor().advise(
        request_id="r1",
        sink_drift_ratio=0.5,
        soft_limit=0.2,
        hard_limit=0.5,
    )
    assert advice.band == "breach"


def test_invalid() -> None:
    """Invalid inputs raise."""

    with pytest.raises(ValueError, match="request_id"):
        AttentionSinkDriftAdvisor().advise(request_id=" ", sink_drift_ratio=0.1)
