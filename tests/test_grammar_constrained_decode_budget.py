"""Unit tests for GrammarConstrainedDecodeBudgetAdvisor."""

from __future__ import annotations

import pytest

from safety.grammar_constrained_decode_budget import GrammarConstrainedDecodeBudgetAdvisor


def test_within() -> None:
    """Low metric is within."""

    advice = GrammarConstrainedDecodeBudgetAdvisor().advise(request_id="r1", constraint_tokens=64.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = GrammarConstrainedDecodeBudgetAdvisor().advise(
        request_id="r1", constraint_tokens=400.0
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = GrammarConstrainedDecodeBudgetAdvisor().advise(
        request_id="r1", constraint_tokens=1200.0
    )
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="constraint_tokens"):
        GrammarConstrainedDecodeBudgetAdvisor().advise(request_id="r1", constraint_tokens=-0.1)
