"""Tests for the latency-slo-shed routing strategy."""

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
    LatencySloShedStrategy,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(request_id="req-slo-shed", messages=[ChatMessage(content="hello")])


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_latency_slo_shed_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("latency-slo-shed") is RoutingStrategyName.LATENCY_SLO_SHED


def test_latency_slo_shed_cold_start_picks_top_quality_model() -> None:
    """With no latency observed yet, every provider is within SLO (best quality)."""
    strategy = LatencySloShedStrategy(
        default_model_catalog(), LatencyStats(), latency_slo_ms=2000.0
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.LATENCY_SLO_SHED
    assert "SLO" in decision.rationale
    assert "shed" in decision.rationale


def test_latency_slo_shed_sheds_slow_providers_when_alternatives_exist() -> None:
    """Over-SLO providers are shed when faster alternatives exist."""
    latency_stats = LatencyStats()
    latency_stats.observe("openai", 2500.0)
    latency_stats.observe("anthropic", 2600.0)
    latency_stats.observe("google", 400.0)
    latency_stats.observe("moonshot", 800.0)
    strategy = LatencySloShedStrategy(default_model_catalog(), latency_stats, latency_slo_ms=2000.0)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "under 2000ms SLO" in decision.rationale
    assert "shed slower alternatives" in decision.rationale


def test_latency_slo_shed_falls_back_to_lowest_latency_when_all_over_slo() -> None:
    """When every provider exceeds the SLO, pick the lowest-p95 model."""
    latency_stats = LatencyStats()
    latency_stats.observe("openai", 3000.0)
    latency_stats.observe("anthropic", 3200.0)
    latency_stats.observe("google", 2800.0)
    latency_stats.observe("moonshot", 2100.0)
    strategy = LatencySloShedStrategy(default_model_catalog(), latency_stats, latency_slo_ms=2000.0)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "no provider under 2000ms SLO" in decision.rationale
    assert "lowest-latency" in decision.rationale


def test_latency_slo_shed_respects_domain_support() -> None:
    """Only medical-capable models are considered for a medical prompt."""
    strategy = LatencySloShedStrategy(
        default_model_catalog(), LatencyStats(), latency_slo_ms=2000.0
    )

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    candidate = default_model_catalog()[decision.chosen_model]
    assert DomainTag.MEDICAL in candidate.supports_domains
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_latency_slo_shed_rejects_negative_slo() -> None:
    """A negative latency SLO fails fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        LatencySloShedStrategy(default_model_catalog(), LatencyStats(), latency_slo_ms=-1.0)


def test_latency_slo_shed_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose latency-slo-shed."""
    catalog = default_model_catalog()
    settings = RouterSettings(latency_slo_ms=1500.0)
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
        latency_slo_ms=settings.latency_slo_ms,
    )

    strategy = strategies[RoutingStrategyName.LATENCY_SLO_SHED]
    assert isinstance(strategy, LatencySloShedStrategy)
    assert strategy.strategy_name is RoutingStrategyName.LATENCY_SLO_SHED
