"""Tests for the provider-family-cost-ceiling routing strategy."""

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
    InflightStats,
    LatencyStats,
    ProviderFamilyCostCeilingStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(request_id="req-pfc", messages=[ChatMessage(content="hello")])


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_provider_family_cost_ceiling_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("provider-family-cost-ceiling")
        is RoutingStrategyName.PROVIDER_FAMILY_COST_CEILING
    )


def test_provider_family_high_ceiling_picks_top_quality_model() -> None:
    """A generous shared ceiling admits every family, so top quality wins."""
    strategy = ProviderFamilyCostCeilingStrategy(
        default_model_catalog(),
        provider_family_cost_ceiling_usd=0.05,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_FAMILY_COST_CEILING
    assert "anthropic family" in decision.rationale
    assert "ceiling" in decision.rationale


def test_provider_family_override_blocks_top_family() -> None:
    """A zero anthropic ceiling forces selection into another family.

    Claude Sonnet 4.6 is the catalog quality leader; blocking the anthropic
    family must fall through to GPT-5.5 (openai) instead of staying on Claude.
    """
    strategy = ProviderFamilyCostCeilingStrategy(
        default_model_catalog(),
        provider_family_cost_ceiling_usd=0.05,
        family_ceilings_usd={"anthropic": 0.0},
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "openai family" in decision.rationale


def test_provider_family_unreachable_ceilings_fall_back_across_families() -> None:
    """When every family ceiling excludes all models, pick the cheapest."""
    strategy = ProviderFamilyCostCeilingStrategy(
        default_model_catalog(),
        provider_family_cost_ceiling_usd=0.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "fell back across families" in decision.rationale


def test_provider_family_respects_domain_support() -> None:
    """Only medical-capable models are considered for a medical prompt."""
    strategy = ProviderFamilyCostCeilingStrategy(
        default_model_catalog(),
        provider_family_cost_ceiling_usd=0.05,
    )

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    candidate = default_model_catalog()[decision.chosen_model]
    assert DomainTag.MEDICAL in candidate.supports_domains
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_provider_family_rejects_negative_default_ceiling() -> None:
    """A negative default cost ceiling fails fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        ProviderFamilyCostCeilingStrategy(
            default_model_catalog(),
            provider_family_cost_ceiling_usd=-0.01,
        )


def test_provider_family_rejects_negative_family_override() -> None:
    """A negative per-family override fails fast at construction."""
    with pytest.raises(ValueError, match="openai"):
        ProviderFamilyCostCeilingStrategy(
            default_model_catalog(),
            provider_family_cost_ceiling_usd=0.05,
            family_ceilings_usd={"openai": -1.0},
        )


def test_provider_family_cost_ceiling_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose provider-family-cost-ceiling."""
    catalog = default_model_catalog()
    settings = RouterSettings(provider_family_cost_ceiling_usd=0.04)
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
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_FAMILY_COST_CEILING]
    assert isinstance(strategy, ProviderFamilyCostCeilingStrategy)
    assert strategy.strategy_name is RoutingStrategyName.PROVIDER_FAMILY_COST_CEILING
