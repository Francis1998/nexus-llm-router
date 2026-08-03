"""Tests for the cache-hit-sticky-warm-pool routing strategy."""

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
    CacheHitStickyWarmPoolStrategy,
    InflightStats,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(*contents: str, session_id: str = "sess-cache") -> RouterRequest:
    """Build a router request with the given message contents."""
    return RouterRequest(
        request_id="req-cache-hit",
        session_id=session_id,
        messages=[ChatMessage(content=content) for content in contents],
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_cache_hit_sticky_warm_pool_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("cache-hit-sticky-warm-pool")
        is RoutingStrategyName.CACHE_HIT_STICKY_WARM_POOL
    )


def test_cache_hit_sticky_warm_pool_is_stable_for_same_prefix() -> None:
    """Identical long prefixes pin to the same model."""
    strategy = CacheHitStickyWarmPoolStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        cache_hit_sticky_min_chars=64,
    )
    prefix = "A" * 80

    first = strategy.choose(_request(prefix, session_id="s1"), _signals())
    second = strategy.choose(_request(prefix, session_id="s2"), _signals())

    assert first.chosen_model == second.chosen_model
    assert "cache-hit-sticky-warm-pool pinned prefix" in first.rationale


def test_cache_hit_sticky_warm_pool_falls_back_to_session_for_short_prefix() -> None:
    """Short prefixes use session_id as the sticky key."""
    strategy = CacheHitStickyWarmPoolStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        cache_hit_sticky_min_chars=64,
    )

    first = strategy.choose(_request("short", session_id="same"), _signals())
    second = strategy.choose(_request("other", session_id="same"), _signals())

    assert first.chosen_model == second.chosen_model


def test_cache_hit_sticky_warm_pool_respects_domain_support() -> None:
    """Only medical-capable models are considered for a medical prompt."""
    strategy = CacheHitStickyWarmPoolStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        cache_hit_sticky_min_chars=64,
    )

    decision = strategy.choose(_request("M" * 80), _signals(DomainTag.MEDICAL))

    candidate = default_model_catalog()[decision.chosen_model]
    assert DomainTag.MEDICAL in candidate.supports_domains


def test_cache_hit_sticky_warm_pool_rejects_non_positive_min_chars() -> None:
    """A non-positive min prefix length fails fast at construction."""
    with pytest.raises(ValueError, match=">= 1"):
        CacheHitStickyWarmPoolStrategy(
            default_model_catalog(),
            CircuitBreakerRegistry(),
            cache_hit_sticky_min_chars=0,
        )


def test_cache_hit_sticky_warm_pool_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose cache-hit-sticky-warm-pool."""
    catalog = default_model_catalog()
    settings = RouterSettings(cache_hit_sticky_min_chars=32)
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
        cache_hit_sticky_min_chars=settings.cache_hit_sticky_min_chars,
    )

    strategy = strategies[RoutingStrategyName.CACHE_HIT_STICKY_WARM_POOL]
    assert isinstance(strategy, CacheHitStickyWarmPoolStrategy)
    assert strategy.strategy_name is RoutingStrategyName.CACHE_HIT_STICKY_WARM_POOL
