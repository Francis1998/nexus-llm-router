"""Tests for the region-carbon-blend routing strategy."""

from __future__ import annotations

import pytest

from router.config import RouterSettings, default_model_catalog
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
    RegionCarbonBlendStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _signals() -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=32,
    )


def _request(metadata: dict[str, str] | None = None) -> RouterRequest:
    """Build a router request with optional carbon metadata."""
    return RouterRequest(
        request_id="req-region-carbon-blend",
        messages=[ChatMessage(content="Route green and fast.")],
        metadata=metadata or {},
    )


def test_region_carbon_blend_enum_parses() -> None:
    """The API header parser can resolve the strategy value."""
    assert RoutingStrategyName("region-carbon-blend") is RoutingStrategyName.REGION_CARBON_BLEND


def test_region_carbon_blend_carbon_only_prefers_lower_intensity() -> None:
    """blend_weight=1 selects the lowest carbon intensity provider."""
    strategy = RegionCarbonBlendStrategy(
        default_model_catalog(),
        LatencyStats(),
        blend_weight=1.0,
    )
    decision = strategy.choose(
        _request(
            {
                "carbon_intensity:openai": "500",
                "carbon_intensity:anthropic": "200",
                "carbon_intensity:google": "450",
                "carbon_intensity:moonshot": "600",
            }
        ),
        _signals(),
    )
    assert decision.provider == "anthropic"
    assert decision.routing_strategy is RoutingStrategyName.REGION_CARBON_BLEND
    assert "intensity 200.0" in decision.rationale
    assert "carbon_weight=1.00" in decision.rationale


def test_region_carbon_blend_latency_only_prefers_lowest_p95() -> None:
    """blend_weight=0 selects the lowest observed p95 provider."""
    latency = LatencyStats()
    latency.observe("anthropic", 900.0)
    latency.observe("openai", 100.0)
    latency.observe("google", 500.0)
    latency.observe("moonshot", 700.0)
    strategy = RegionCarbonBlendStrategy(
        default_model_catalog(),
        latency,
        blend_weight=0.0,
    )
    decision = strategy.choose(
        _request(
            {
                "carbon_intensity:openai": "900",
                "carbon_intensity:anthropic": "100",
                "carbon_intensity:google": "200",
                "carbon_intensity:moonshot": "150",
            }
        ),
        _signals(),
    )
    assert decision.provider == "openai"
    assert "carbon_weight=0.00" in decision.rationale
    assert "p95 100.0ms" in decision.rationale


def test_region_carbon_blend_balances_carbon_and_latency() -> None:
    """A mid blend can pick a moderate carbon provider with better latency."""
    latency = LatencyStats()
    latency.observe("anthropic", 800.0)
    latency.observe("openai", 120.0)
    latency.observe("google", 150.0)
    latency.observe("moonshot", 700.0)
    strategy = RegionCarbonBlendStrategy(
        default_model_catalog(),
        latency,
        blend_weight=0.5,
    )
    decision = strategy.choose(
        _request(
            {
                "carbon_intensity:openai": "400",
                "carbon_intensity:anthropic": "100",
                "carbon_intensity:google": "390",
                "carbon_intensity:moonshot": "500",
            }
        ),
        _signals(),
    )
    assert decision.provider == "openai"
    assert "region-carbon-blend selected" in decision.rationale


def test_region_carbon_blend_region_heuristic() -> None:
    """Missing intensity metadata falls back to region defaults."""
    strategy = RegionCarbonBlendStrategy(
        default_model_catalog(),
        LatencyStats(),
        blend_weight=1.0,
    )
    decision = strategy.choose(_request({"region": "eu"}), _signals())
    assert "region-carbon-blend selected" in decision.rationale
    assert "250.0" in decision.rationale


def test_region_carbon_blend_rejects_invalid_weight() -> None:
    """blend_weight outside [0, 1] fails fast."""
    with pytest.raises(ValueError, match="blend_weight"):
        RegionCarbonBlendStrategy(default_model_catalog(), LatencyStats(), blend_weight=1.5)


def test_region_carbon_blend_registered_by_strategy_factory() -> None:
    """The built-in strategy map exposes region-carbon-blend."""
    settings = RouterSettings(region_carbon_blend_weight=0.7)
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
        region_carbon_blend_weight=settings.region_carbon_blend_weight,
    )
    strategy = strategies[RoutingStrategyName.REGION_CARBON_BLEND]
    assert isinstance(strategy, RegionCarbonBlendStrategy)
    assert strategy._blend_weight == 0.7  # noqa: SLF001


def test_region_carbon_blend_settings_default() -> None:
    """RouterSettings expose the carbon blend weight default."""
    assert RouterSettings().region_carbon_blend_weight == 0.5
