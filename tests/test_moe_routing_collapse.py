"""Tests for MoeRoutingCollapseAdvisor."""

from __future__ import annotations

import pytest

from safety.moe_routing_collapse import MoeRoutingCollapseAdvisor


def test_within() -> None:
    """Band within."""

    advice = MoeRoutingCollapseAdvisor().advise(request_id="r1", collapse_rate=0.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.2 + 0.5) / 2
    advice = MoeRoutingCollapseAdvisor().advise(request_id="r1", collapse_rate=mid)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = MoeRoutingCollapseAdvisor().advise(request_id="r1", collapse_rate=1.0000)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError, match="request_id"):
        MoeRoutingCollapseAdvisor().advise(request_id=" ", collapse_rate=0.0)
