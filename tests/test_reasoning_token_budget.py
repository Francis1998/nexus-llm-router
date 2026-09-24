"""Unit tests for ReasoningTokenBudgetAdvisor."""

from __future__ import annotations

import pytest

from safety.reasoning_token_budget import ReasoningTokenBudgetAdvisor


def test_ok_band() -> None:
    """Plenty of reasoning budget remaining is ok."""

    advice = ReasoningTokenBudgetAdvisor().advise(
        reasoning_tokens_used=100,
        reasoning_token_budget=2000,
    )
    assert advice.band == "ok"
    assert advice.remaining == 1900


def test_near_limit() -> None:
    """Within 10% remaining is near_limit."""

    advice = ReasoningTokenBudgetAdvisor().advise(
        reasoning_tokens_used=1900,
        reasoning_token_budget=2000,
    )
    assert advice.band == "near_limit"


def test_exhausted() -> None:
    """Zero remaining is exhausted."""

    advice = ReasoningTokenBudgetAdvisor().advise(
        reasoning_tokens_used=2000,
        reasoning_token_budget=2000,
    )
    assert advice.band == "exhausted"


def test_invalid_budget_raises() -> None:
    """Non-positive budget raises ValueError."""

    with pytest.raises(ValueError, match="reasoning_token_budget"):
        ReasoningTokenBudgetAdvisor().advise(
            reasoning_tokens_used=0,
            reasoning_token_budget=0,
        )
