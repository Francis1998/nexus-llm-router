"""Unit tests for SpeculativeDecodeAbortAdvisor."""

from __future__ import annotations

import pytest

from safety.speculative_decode_abort import SpeculativeDecodeAbortAdvisor


def test_healthy_band() -> None:
    """At/under budget is healthy."""

    advice = SpeculativeDecodeAbortAdvisor().advise(
        request_id="r1", abort_rate=0.05, budget_rate=0.1
    )
    assert advice.band == "healthy"


def test_elevated_band() -> None:
    """Up to 1.5x budget is elevated."""

    advice = SpeculativeDecodeAbortAdvisor().advise(
        request_id="r1", abort_rate=0.12, budget_rate=0.1
    )
    assert advice.band == "elevated"


def test_critical_band() -> None:
    """Over 1.5x budget is critical."""

    advice = SpeculativeDecodeAbortAdvisor().advise(
        request_id="r1", abort_rate=0.2, budget_rate=0.1
    )
    assert advice.band == "critical"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        SpeculativeDecodeAbortAdvisor().advise(request_id=" ", abort_rate=0.1, budget_rate=0.1)
