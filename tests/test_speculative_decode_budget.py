"""Unit tests for SpeculativeDecodeBudgetAdvisor."""

from __future__ import annotations

import pytest

from safety.speculative_decode_budget import SpeculativeDecodeBudgetAdvisor


def test_ok_band() -> None:
    """Low draft utilization is ok."""

    advice = SpeculativeDecodeBudgetAdvisor().advise(
        request_id="r1",
        draft_tokens=10,
        max_draft_tokens=64,
    )
    assert advice.band == "ok"


def test_elevated_band() -> None:
    """Mid draft utilization is elevated."""

    advice = SpeculativeDecodeBudgetAdvisor().advise(
        request_id="r1",
        draft_tokens=50,
        max_draft_tokens=64,
    )
    assert advice.band == "elevated"


def test_saturated_band() -> None:
    """Full draft budget is saturated."""

    advice = SpeculativeDecodeBudgetAdvisor().advise(
        request_id="r1",
        draft_tokens=64,
        max_draft_tokens=64,
    )
    assert advice.band == "saturated"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        SpeculativeDecodeBudgetAdvisor().advise(request_id=" ", draft_tokens=1)
