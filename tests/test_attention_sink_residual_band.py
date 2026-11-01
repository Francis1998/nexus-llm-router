"""Tests for AttentionSinkResidualBandAdvisor."""

from __future__ import annotations

import pytest

from safety.attention_sink_residual_band import AttentionSinkResidualBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = AttentionSinkResidualBandAdvisor().advise(request_id="r1", sink_residual_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = AttentionSinkResidualBandAdvisor().advise(request_id="r1", sink_residual_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = AttentionSinkResidualBandAdvisor().advise(request_id="r1", sink_residual_ratio=0.9)
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        AttentionSinkResidualBandAdvisor().advise(request_id=" ", sink_residual_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="sink_residual_ratio"):
        AttentionSinkResidualBandAdvisor().advise(request_id="r1", sink_residual_ratio=-0.1)
