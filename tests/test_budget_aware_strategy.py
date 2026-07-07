"""Tests for the budget-aware routing strategy."""

import pytest

from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    OPENAI_BALANCED_MODEL,
)
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import BudgetAwareStrategy


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(request_id="req-1", messages=[ChatMessage(content="hello")])


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_budget_aware_high_ceiling_picks_top_quality_model() -> None:
    """A generous ceiling admits every model, so the best quality is chosen."""
    strategy = BudgetAwareStrategy(default_model_catalog(), request_cost_ceiling_usd=0.05)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy == RoutingStrategyName.BUDGET_AWARE
    assert "ceiling" in decision.rationale


def test_budget_aware_tight_ceiling_picks_best_affordable_model() -> None:
    """A tight ceiling admits only cheap models; the best affordable one wins.

    With this prompt the estimated cost of the balanced OpenAI model
    (~$0.0004) and the Moonshot model (~$0.0010) both fit a $0.0015 ceiling,
    while every higher-quality model exceeds it. The balanced OpenAI model has
    the higher quality of the two affordable candidates.
    """
    strategy = BudgetAwareStrategy(default_model_catalog(), request_cost_ceiling_usd=0.0015)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL


def test_budget_aware_unreachable_ceiling_falls_back_to_cheapest() -> None:
    """A zero ceiling admits nothing, so the cheapest eligible model is used."""
    strategy = BudgetAwareStrategy(default_model_catalog(), request_cost_ceiling_usd=0.0)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "no model within" in decision.rationale


def test_budget_aware_respects_domain_support() -> None:
    """Only medical-capable models are considered for a medical prompt."""
    strategy = BudgetAwareStrategy(default_model_catalog(), request_cost_ceiling_usd=0.05)

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    candidate = default_model_catalog()[decision.chosen_model]
    assert DomainTag.MEDICAL in candidate.supports_domains
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_budget_aware_rejects_negative_ceiling() -> None:
    """A negative cost ceiling fails fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        BudgetAwareStrategy(default_model_catalog(), request_cost_ceiling_usd=-0.01)
