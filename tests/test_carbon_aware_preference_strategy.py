"""Tests for the carbon-aware-preference routing strategy."""

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
    CarbonAwarePreferenceStrategy,
    InflightStats,
    LatencyStats,
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
        request_id="req-carbon",
        messages=[ChatMessage(content="Route green.")],
        metadata=metadata or {},
    )


def test_carbon_aware_preference_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("carbon-aware-preference")
        is RoutingStrategyName.CARBON_AWARE_PREFERENCE
    )


def test_carbon_aware_preference_prefers_lower_intensity() -> None:
    """Lower carbon_intensity metadata wins under the max intensity."""
    strategy = CarbonAwarePreferenceStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        max_intensity=400.0,
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
    assert "intensity 200.0" in decision.rationale


def test_carbon_aware_preference_falls_back_when_all_over_cap() -> None:
    """When all intensities exceed the cap, still pick the lowest intensity."""
    strategy = CarbonAwarePreferenceStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        max_intensity=100.0,
    )
    decision = strategy.choose(
        _request(
            {
                "carbon_intensity:openai": "500",
                "carbon_intensity:anthropic": "300",
                "carbon_intensity:google": "450",
                "carbon_intensity:moonshot": "600",
            }
        ),
        _signals(),
    )
    assert decision.provider == "anthropic"


def test_carbon_aware_preference_region_heuristic() -> None:
    """Missing intensity metadata falls back to region defaults."""
    strategy = CarbonAwarePreferenceStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        max_intensity=400.0,
    )
    decision = strategy.choose(_request({"region": "eu"}), _signals())
    assert "carbon-aware-preference selected" in decision.rationale
    assert "250.0" in decision.rationale


def test_carbon_aware_preference_rejects_negative_max() -> None:
    """Negative max intensity fails fast."""
    with pytest.raises(ValueError, match="max_intensity"):
        CarbonAwarePreferenceStrategy(
            default_model_catalog(),
            CircuitBreakerRegistry(),
            max_intensity=-1.0,
        )


def test_carbon_aware_preference_registered_by_strategy_factory() -> None:
    """The built-in strategy map exposes carbon-aware-preference."""
    settings = RouterSettings(carbon_aware_max_intensity=350.0)
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
        embedding_cache_namespace_prefix=settings.embedding_cache_namespace_prefix,
        circuit_half_open_probe_budget=settings.circuit_half_open_probe_budget,
        carbon_aware_max_intensity=settings.carbon_aware_max_intensity,
    )
    strategy = strategies[RoutingStrategyName.CARBON_AWARE_PREFERENCE]
    assert isinstance(strategy, CarbonAwarePreferenceStrategy)
    assert strategy._max_intensity == 350.0  # noqa: SLF001


def test_carbon_aware_preference_settings_default() -> None:
    """RouterSettings expose the max intensity default."""
    assert RouterSettings().carbon_aware_max_intensity == 400.0
