"""Tests for SpeculativeAcceptanceCollapseBandAdvisor."""

from __future__ import annotations

import pytest

from safety.speculative_acceptance_collapse_band import SpeculativeAcceptanceCollapseBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = SpeculativeAcceptanceCollapseBandAdvisor().advise(request_id="r1", collapse_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = SpeculativeAcceptanceCollapseBandAdvisor().advise(request_id="r1", collapse_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = SpeculativeAcceptanceCollapseBandAdvisor().advise(request_id="r1", collapse_ratio=0.9)
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        SpeculativeAcceptanceCollapseBandAdvisor().advise(request_id=" ", collapse_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="collapse_ratio"):
        SpeculativeAcceptanceCollapseBandAdvisor().advise(request_id="r1", collapse_ratio=-0.1)
