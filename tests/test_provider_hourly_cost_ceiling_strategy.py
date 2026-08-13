"""Tests for the provider-hourly-cost-ceiling routing strategy."""

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
    ProviderHourlyCostCeilingStrategy,
    ProviderHourlySpendWindow,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(request_id="req-phc", messages=[ChatMessage(content="hello")])


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_provider_hourly_cost_ceiling_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("provider-hourly-cost-ceiling")
        is RoutingStrategyName.PROVIDER_HOURLY_COST_CEILING
    )


def test_provider_hourly_under_ceiling_picks_top_quality() -> None:
    """With no recorded spend, every provider is under ceiling and quality wins."""
    strategy = ProviderHourlyCostCeilingStrategy(
        default_model_catalog(),
        ProviderHourlySpendWindow(),
        provider_hourly_cost_ceiling_usd=5.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_HOURLY_COST_CEILING
    assert "under" in decision.rationale
    assert "ceiling" in decision.rationale


def test_provider_hourly_skips_over_ceiling_provider() -> None:
    """An over-ceiling anthropic provider forces GPT-5.5 (openai) selection.

    Claude Sonnet 4.6 leads quality; blocking anthropic hourly spend must fall
    through to the next quality leader instead of staying on Claude.
    """
    spend = ProviderHourlySpendWindow()
    spend.record("anthropic", 6.0)
    strategy = ProviderHourlyCostCeilingStrategy(
        default_model_catalog(),
        spend,
        provider_hourly_cost_ceiling_usd=5.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "under" in decision.rationale


def test_provider_hourly_all_over_falls_back_to_cheapest() -> None:
    """When every provider is over the hourly ceiling, pick the cheapest."""
    spend = ProviderHourlySpendWindow()
    for provider in ("anthropic", "openai", "google", "moonshot"):
        spend.record(provider, 10.0)
    strategy = ProviderHourlyCostCeilingStrategy(
        default_model_catalog(),
        spend,
        provider_hourly_cost_ceiling_usd=5.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "every provider over" in decision.rationale


def test_provider_hourly_respects_domain_support() -> None:
    """Only medical-capable models are considered for a medical prompt."""
    strategy = ProviderHourlyCostCeilingStrategy(
        default_model_catalog(),
        ProviderHourlySpendWindow(),
        provider_hourly_cost_ceiling_usd=5.0,
    )

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    candidate = default_model_catalog()[decision.chosen_model]
    assert DomainTag.MEDICAL in candidate.supports_domains
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_provider_hourly_rejects_negative_ceiling() -> None:
    """A negative hourly ceiling fails fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        ProviderHourlyCostCeilingStrategy(
            default_model_catalog(),
            ProviderHourlySpendWindow(),
            provider_hourly_cost_ceiling_usd=-0.01,
        )


def test_provider_hourly_spend_window_prunes_outside_hour() -> None:
    """Spend older than the rolling hour must not count toward the ceiling."""
    spend = ProviderHourlySpendWindow(window_seconds=3600.0)
    spend.record("openai", 4.0, now=0.0)
    spend.record("openai", 3.0, now=4000.0)

    assert spend.provider_spend("openai", now=4000.0) == pytest.approx(3.0)
    assert not spend.is_over_ceiling("openai", 5.0, now=4000.0)


def test_provider_hourly_cost_ceiling_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose provider-hourly-cost-ceiling."""
    catalog = default_model_catalog()
    settings = RouterSettings(provider_hourly_cost_ceiling_usd=4.0)
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
        provider_hourly_cost_ceiling_usd=settings.provider_hourly_cost_ceiling_usd,
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_HOURLY_COST_CEILING]
    assert isinstance(strategy, ProviderHourlyCostCeilingStrategy)
    assert strategy.strategy_name is RoutingStrategyName.PROVIDER_HOURLY_COST_CEILING
