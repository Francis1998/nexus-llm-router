"""Tests for the multi-region-latency-hedge routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    ModelCandidate,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    InflightStats,
    LatencyStats,
    MultiRegionLatencyHedgeStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def _request(region: str = "us") -> RouterRequest:
    """Build a router request with a region affinity."""
    return RouterRequest(
        request_id="req-latency-hedge",
        messages=[ChatMessage(content="hello")],
        region=region,
    )


def _catalog() -> dict[str, ModelCandidate]:
    """Build a two-region catalog for hedge tests."""
    return {
        "us-premium": ModelCandidate(
            model="us-premium",
            provider="openai",
            quality_score=0.95,
            input_cost_per_1k=0.006,
            output_cost_per_1k=0.018,
            supports_domains={DomainTag.GENERAL},
            supported_regions={"us"},
        ),
        "cn-fast": ModelCandidate(
            model="cn-fast",
            provider="moonshot",
            quality_score=0.76,
            input_cost_per_1k=0.0005,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
            supported_regions={"cn"},
        ),
    }


def test_multi_region_latency_hedge_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("multi-region-latency-hedge")
        is RoutingStrategyName.MULTI_REGION_LATENCY_HEDGE
    )


def test_multi_region_latency_hedge_cold_start_stays_on_primary_quality() -> None:
    """With no latency observed yet, routing stays on the primary region pick."""
    strategy = MultiRegionLatencyHedgeStrategy(_catalog(), LatencyStats(), latency_hedge_ms=500.0)

    decision = strategy.choose(_request("us"), _signals())

    assert decision.chosen_model == "us-premium"
    assert decision.routing_strategy is RoutingStrategyName.MULTI_REGION_LATENCY_HEDGE
    assert "stayed on primary region" in decision.rationale


def test_multi_region_latency_hedge_hedges_to_secondary_when_primary_hot() -> None:
    """A hot primary region should hedge to the lowest-p50 secondary candidate."""
    latency_stats = LatencyStats()
    latency_stats.observe("openai", 900.0)
    strategy = MultiRegionLatencyHedgeStrategy(_catalog(), latency_stats, latency_hedge_ms=500.0)

    decision = strategy.choose(_request("us"), _signals())

    assert decision.chosen_model == "cn-fast"
    assert "hedged to secondary" in decision.rationale


def test_multi_region_latency_hedge_stays_primary_when_under_threshold() -> None:
    """Primary quality should win when provider p50 is within the hedge threshold."""
    latency_stats = LatencyStats()
    latency_stats.observe("openai", 300.0)
    latency_stats.observe("moonshot", 100.0)
    strategy = MultiRegionLatencyHedgeStrategy(_catalog(), latency_stats, latency_hedge_ms=500.0)

    decision = strategy.choose(_request("us"), _signals())

    assert decision.chosen_model == "us-premium"
    assert "stayed on primary region" in decision.rationale


def test_multi_region_latency_hedge_picks_fastest_secondary_on_tie_break() -> None:
    """Secondary hedging should prefer the lowest observed p50."""
    catalog = {
        **_catalog(),
        "eu-fast": ModelCandidate(
            model="eu-fast",
            provider="google",
            quality_score=0.90,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.01,
            supports_domains={DomainTag.GENERAL},
            supported_regions={"eu"},
        ),
    }
    latency_stats = LatencyStats()
    latency_stats.observe("openai", 800.0)
    latency_stats.observe("moonshot", 250.0)
    latency_stats.observe("google", 120.0)
    strategy = MultiRegionLatencyHedgeStrategy(catalog, latency_stats, latency_hedge_ms=500.0)

    decision = strategy.choose(_request("us"), _signals())

    assert decision.chosen_model == "eu-fast"
    assert "hedged to secondary" in decision.rationale


def test_multi_region_latency_hedge_respects_domain_support() -> None:
    """Only medical-capable models should be considered for medical prompts."""
    catalog = {
        "us-medical": ModelCandidate(
            model="us-medical",
            provider="anthropic",
            quality_score=0.98,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            supports_domains={DomainTag.MEDICAL},
            supported_regions={"us"},
        ),
        "cn-general": ModelCandidate(
            model="cn-general",
            provider="moonshot",
            quality_score=0.76,
            input_cost_per_1k=0.0005,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
            supported_regions={"cn"},
        ),
    }
    latency_stats = LatencyStats()
    latency_stats.observe("anthropic", 900.0)
    strategy = MultiRegionLatencyHedgeStrategy(catalog, latency_stats, latency_hedge_ms=500.0)

    decision = strategy.choose(_request("us"), _signals(DomainTag.MEDICAL))

    assert decision.chosen_model == "us-medical"
    assert "stayed on primary region" in decision.rationale


def test_multi_region_latency_hedge_rejects_negative_threshold() -> None:
    """A negative hedge threshold fails fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        MultiRegionLatencyHedgeStrategy(_catalog(), LatencyStats(), latency_hedge_ms=-1.0)


def test_multi_region_latency_hedge_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose multi-region-latency-hedge."""
    catalog = default_model_catalog()
    settings = RouterSettings(latency_hedge_ms=450.0)
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
        latency_hedge_ms=settings.latency_hedge_ms,
    )

    strategy = strategies[RoutingStrategyName.MULTI_REGION_LATENCY_HEDGE]
    assert isinstance(strategy, MultiRegionLatencyHedgeStrategy)
    assert strategy.strategy_name is RoutingStrategyName.MULTI_REGION_LATENCY_HEDGE
