"""Tests for the region-latency-p99-shed routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_PRO_MODEL,
    MOONSHOT_BALANCED_MODEL,
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
    RegionLatencyP99ShedStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(region: str | None = None) -> RouterRequest:
    """Build a minimal router request with optional region affinity."""
    return RouterRequest(
        request_id="req-region-p99",
        messages=[ChatMessage(content="Shed hot regional tails.")],
        region=region,
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=64,
    )


def _strategy(
    *,
    latency_stats: LatencyStats | None = None,
    region_latency_p99_ms: float = 3000.0,
) -> RegionLatencyP99ShedStrategy:
    """Build region-latency-p99-shed with overridable dependencies."""
    return RegionLatencyP99ShedStrategy(
        default_model_catalog(),
        latency_stats or LatencyStats(),
        region_latency_p99_ms=region_latency_p99_ms,
    )


def test_region_latency_p99_shed_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("region-latency-p99-shed")
        is RoutingStrategyName.REGION_LATENCY_P99_SHED
    )


def test_latency_stats_p99_cold_start_is_zero() -> None:
    """Providers with no observations report p99 0.0 for cold starts."""
    assert LatencyStats().p99("openai") == 0.0


def test_region_latency_p99_shed_cold_start_picks_top_quality() -> None:
    """Cold LatencyStats treat every provider as under the p99 threshold."""
    strategy = _strategy()

    decision = strategy.choose(_request("eu"), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.REGION_LATENCY_P99_SHED
    assert "under 3000ms p99" in decision.rationale
    assert "region 'eu'" in decision.rationale


def test_region_latency_p99_shed_sheds_hot_regional_providers() -> None:
    """Over-p99 regional providers are shed when faster regional options exist."""
    latency_stats = LatencyStats()
    latency_stats.observe("anthropic", 4500.0)
    latency_stats.observe("google", 800.0)
    latency_stats.observe("openai", 5000.0)
    strategy = _strategy(latency_stats=latency_stats)

    decision = strategy.choose(_request("eu"), _signals())

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "under 3000ms p99" in decision.rationale
    assert "shed slower alternatives" in decision.rationale


def test_region_latency_p99_shed_falls_back_to_lowest_p99() -> None:
    """When every regional provider exceeds p99, pick the lowest-p99 model."""
    latency_stats = LatencyStats()
    latency_stats.observe("anthropic", 5200.0)
    latency_stats.observe("google", 4800.0)
    latency_stats.observe("openai", 6100.0)
    latency_stats.observe("moonshot", 4100.0)
    strategy = _strategy(latency_stats=latency_stats, region_latency_p99_ms=3000.0)

    decision = strategy.choose(_request("global"), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "no provider in region 'global' under 3000ms p99" in decision.rationale
    assert "lowest-p99" in decision.rationale


def test_region_latency_p99_shed_rejects_negative_threshold() -> None:
    """A negative p99 threshold fails fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        _strategy(region_latency_p99_ms=-1.0)


def test_region_latency_p99_shed_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose region-latency-p99-shed."""
    settings = RouterSettings(region_latency_p99_ms=2500.0)
    strategies = build_strategies(
        default_model_catalog(),
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
        cache_hit_sticky_min_chars=settings.cache_hit_sticky_min_chars,
        tenant_concurrency_lease=settings.tenant_concurrency_lease,
        provider_error_budget_rate=settings.provider_error_budget_rate,
        region_latency_p99_ms=settings.region_latency_p99_ms,
    )

    strategy = strategies[RoutingStrategyName.REGION_LATENCY_P99_SHED]
    assert isinstance(strategy, RegionLatencyP99ShedStrategy)
    assert strategy.strategy_name is RoutingStrategyName.REGION_LATENCY_P99_SHED


def test_region_latency_p99_shed_settings_default() -> None:
    """RouterSettings expose the region p99 threshold default."""
    assert RouterSettings().region_latency_p99_ms == 3000.0
