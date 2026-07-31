"""Tests for the soft-family-budget routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    OPENAI_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
)
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    FamilySpendWindow,
    InflightStats,
    LatencyStats,
    SoftFamilyBudgetStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(request_id="req-sfb", messages=[ChatMessage(content="hello")])


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_soft_family_budget_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("soft-family-budget") is RoutingStrategyName.SOFT_FAMILY_BUDGET


def test_soft_family_budget_prefers_highest_quality_under_budget() -> None:
    """Families under the soft budget admit the highest-quality eligible model."""
    strategy = SoftFamilyBudgetStrategy(
        default_model_catalog(),
        FamilySpendWindow(),
        soft_family_budget_usd=5.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.SOFT_FAMILY_BUDGET
    assert "under" in decision.rationale
    assert "soft budget" in decision.rationale


def test_soft_family_budget_skips_over_budget_family() -> None:
    """An over-budget anthropic family should route to another under-budget family."""
    spend_window = FamilySpendWindow(window_seconds=3600.0)
    spend_window.record("anthropic", 6.0)
    strategy = SoftFamilyBudgetStrategy(
        default_model_catalog(),
        spend_window,
        soft_family_budget_usd=5.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert decision.chosen_model != ANTHROPIC_SAFETY_MODEL
    assert "under" in decision.rationale


def test_soft_family_budget_falls_back_to_cheapest_other_family() -> None:
    """When every family is over budget, pick the cheapest other family."""
    spend_window = FamilySpendWindow(window_seconds=3600.0)
    for family in ("anthropic", "openai", "google", "moonshot"):
        spend_window.record(family, 10.0)
    strategy = SoftFamilyBudgetStrategy(
        default_model_catalog(),
        spend_window,
        soft_family_budget_usd=5.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "cheapest other family" in decision.rationale


def test_soft_family_budget_respects_domain_support() -> None:
    """Medical prompts still require domain-eligible models."""
    strategy = SoftFamilyBudgetStrategy(
        default_model_catalog(),
        FamilySpendWindow(),
        soft_family_budget_usd=5.0,
    )

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    candidate = default_model_catalog()[decision.chosen_model]
    assert DomainTag.MEDICAL in candidate.supports_domains
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_soft_family_budget_rejects_negative_budget() -> None:
    """A negative soft budget fails fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        SoftFamilyBudgetStrategy(
            default_model_catalog(),
            FamilySpendWindow(),
            soft_family_budget_usd=-0.01,
        )


def test_family_spend_window_prunes_outside_rolling_window() -> None:
    """Spend older than the window must not count toward the family total."""
    spend_window = FamilySpendWindow(window_seconds=60.0)
    spend_window.record("openai", 4.0, now=0.0)
    spend_window.record("openai", 3.0, now=120.0)

    assert spend_window.family_spend("openai", now=120.0) == pytest.approx(3.0)
    assert not spend_window.is_over_budget("openai", 5.0, now=120.0)


def test_soft_family_budget_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose soft-family-budget."""
    catalog = default_model_catalog()
    settings = RouterSettings(soft_family_budget_usd=4.0)
    spend_window = FamilySpendWindow(settings.soft_family_budget_window_seconds)
    strategies = build_strategies(
        catalog,
        LatencyStats(),
        InflightStats(),
        settings.quality_floor,
        settings.ab_model_a,
        settings.ab_model_b,
        settings.ab_model_a_weight,
        CircuitBreakerRegistry(),
        settings.blend_quality_weight,
        settings.blend_cost_weight,
        settings.blend_latency_weight,
        settings.request_cost_ceiling_usd,
        settings.canary_stable_model,
        settings.canary_model,
        settings.canary_weight,
        settings.latency_sla_ms,
        settings.prompt_prefix_cache_min_chars,
        settings.epsilon,
        settings.availability_slo,
        SuccessStats(),
        settings.failover_priority,
        settings.health_blend_success_weight,
        settings.health_blend_latency_weight,
        settings.health_blend_quality_weight,
        settings.health_blend_cost_weight,
        settings.concurrency_cap,
        provider_family_cost_ceiling_usd=settings.provider_family_cost_ceiling_usd,
        family_spend_window=spend_window,
        soft_family_budget_usd=settings.soft_family_budget_usd,
    )

    strategy = strategies[RoutingStrategyName.SOFT_FAMILY_BUDGET]
    assert isinstance(strategy, SoftFamilyBudgetStrategy)
    assert strategy.strategy_name is RoutingStrategyName.SOFT_FAMILY_BUDGET
