"""Tests for ExpertRoutingEntropyBandAdvisor."""

from __future__ import annotations

import pytest

from safety.expert_routing_entropy_band import ExpertRoutingEntropyBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = ExpertRoutingEntropyBandAdvisor().advise(request_id="r1", entropy_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = ExpertRoutingEntropyBandAdvisor().advise(request_id="r1", entropy_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = ExpertRoutingEntropyBandAdvisor().advise(request_id="r1", entropy_ratio=0.9)
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        ExpertRoutingEntropyBandAdvisor().advise(request_id=" ", entropy_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="entropy_ratio"):
        ExpertRoutingEntropyBandAdvisor().advise(request_id="r1", entropy_ratio=-0.1)
