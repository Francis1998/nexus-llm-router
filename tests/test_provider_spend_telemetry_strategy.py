"""Tests for the provider-spend-telemetry routing strategy."""

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
    ProviderSpendTelemetryStrategy,
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
        prompt_tokens_estimate=32,
    )


def _request(metadata: dict[str, str] | None = None) -> RouterRequest:
    """Build a router request with optional spend metadata."""
    return RouterRequest(
        request_id="req-spend",
        messages=[ChatMessage(content="Route by spend.")],
        metadata=metadata or {},
    )


def test_provider_spend_telemetry_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("provider-spend-telemetry")
        is RoutingStrategyName.PROVIDER_SPEND_TELEMETRY
    )


def test_provider_spend_telemetry_under_threshold_uses_quality() -> None:
    """When all spend is under the soft threshold, quality wins."""
    strategy = ProviderSpendTelemetryStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        soft_spend_usd=10.0,
    )
    decision = strategy.choose(
        _request({"spend:openai": "1.0", "spend:anthropic": "2.0"}),
        _signals(),
    )
    assert "under soft spend" in decision.rationale


def test_provider_spend_telemetry_prefers_lower_spend_when_over_threshold() -> None:
    """Once spend exceeds the soft threshold, lower-spend providers win."""
    strategy = ProviderSpendTelemetryStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        soft_spend_usd=5.0,
    )
    decision = strategy.choose(
        _request(
            {
                "spend:openai": "40.0",
                "spend:anthropic": "6.0",
                "spend:google": "50.0",
                "spend:moonshot": "55.0",
            }
        ),
        _signals(),
    )
    assert decision.provider == "anthropic"
    assert "preferred lower-spend provider anthropic" in decision.rationale


def test_provider_spend_telemetry_rejects_negative_soft_spend() -> None:
    """Negative soft spend thresholds fail fast."""
    with pytest.raises(ValueError, match="soft_spend_usd"):
        ProviderSpendTelemetryStrategy(
            default_model_catalog(),
            CircuitBreakerRegistry(),
            soft_spend_usd=-1.0,
        )


def test_provider_spend_telemetry_ignores_unparseable_spend() -> None:
    """Unparseable spend metadata is treated as zero."""
    strategy = ProviderSpendTelemetryStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        soft_spend_usd=10.0,
    )
    decision = strategy.choose(_request({"spend:openai": "not-a-number"}), _signals())
    assert "under soft spend" in decision.rationale


def test_provider_spend_telemetry_registered_by_strategy_factory() -> None:
    """The built-in strategy map exposes provider-spend-telemetry."""
    settings = RouterSettings(provider_spend_soft_usd=7.5)
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
        provider_spend_soft_usd=settings.provider_spend_soft_usd,
    )
    strategy = strategies[RoutingStrategyName.PROVIDER_SPEND_TELEMETRY]
    assert isinstance(strategy, ProviderSpendTelemetryStrategy)
    assert strategy._soft_spend_usd == 7.5  # noqa: SLF001


def test_provider_spend_telemetry_settings_default() -> None:
    """RouterSettings expose the soft spend default."""
    assert RouterSettings().provider_spend_soft_usd == 10.0
