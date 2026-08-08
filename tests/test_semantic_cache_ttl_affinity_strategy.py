"""Tests for the semantic-cache-ttl-affinity routing strategy."""

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
    SemanticCacheTtlAffinityStrategy,
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
    """Build a router request with optional cache metadata."""
    return RouterRequest(
        request_id="req-cache-ttl",
        messages=[ChatMessage(content="Use warm cache.")],
        metadata=metadata or {},
    )


def test_semantic_cache_ttl_affinity_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("semantic-cache-ttl-affinity")
        is RoutingStrategyName.SEMANTIC_CACHE_TTL_AFFINITY
    )


def test_semantic_cache_ttl_affinity_pins_warm_provider() -> None:
    """Cacheable requests pin to the warmest in-window provider."""
    strategy = SemanticCacheTtlAffinityStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        ttl_seconds=300.0,
    )
    decision = strategy.choose(
        _request(
            {
                "cacheable": "true",
                "cache_ttl_remaining:openai": "10",
                "cache_ttl_remaining:anthropic": "250",
                "cache_ttl_remaining:google": "40",
                "cache_ttl_remaining:moonshot": "5",
            }
        ),
        _signals(),
    )
    assert decision.provider == "anthropic"
    assert "pinned warm provider anthropic" in decision.rationale


def test_semantic_cache_ttl_affinity_fallback_when_not_cacheable() -> None:
    """Non-cacheable requests fall back to quality routing."""
    strategy = SemanticCacheTtlAffinityStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        ttl_seconds=300.0,
    )
    decision = strategy.choose(
        _request({"cache_ttl_remaining:anthropic": "250"}),
        _signals(),
    )
    assert "fallback quality route" in decision.rationale


def test_semantic_cache_ttl_affinity_ignores_ttl_above_window() -> None:
    """TTL values above the configured window are not treated as warm."""
    strategy = SemanticCacheTtlAffinityStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        ttl_seconds=100.0,
    )
    decision = strategy.choose(
        _request(
            {
                "cacheable": "true",
                "cache_ttl_remaining:openai": "250",
                "cache_ttl_remaining:anthropic": "250",
            }
        ),
        _signals(),
    )
    assert "fallback quality route" in decision.rationale


def test_semantic_cache_ttl_affinity_rejects_negative_ttl() -> None:
    """Negative TTL windows fail fast."""
    with pytest.raises(ValueError, match="ttl_seconds"):
        SemanticCacheTtlAffinityStrategy(
            default_model_catalog(),
            CircuitBreakerRegistry(),
            ttl_seconds=-1.0,
        )


def test_semantic_cache_ttl_affinity_registered_by_strategy_factory() -> None:
    """The built-in strategy map exposes semantic-cache-ttl-affinity."""
    settings = RouterSettings(semantic_cache_ttl_seconds=120.0)
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
        semantic_cache_ttl_seconds=settings.semantic_cache_ttl_seconds,
    )
    strategy = strategies[RoutingStrategyName.SEMANTIC_CACHE_TTL_AFFINITY]
    assert isinstance(strategy, SemanticCacheTtlAffinityStrategy)
    assert strategy._ttl_seconds == 120.0  # noqa: SLF001


def test_semantic_cache_ttl_affinity_settings_default() -> None:
    """RouterSettings expose the TTL default."""
    assert RouterSettings().semantic_cache_ttl_seconds == 300.0
