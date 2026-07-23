"""Tests for the provider-health-score-blend routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_FRONTIER_MODEL
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
    LatencyStats,
    ProviderHealthScoreBlendStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(max_tokens: int = 512) -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id="req-health-blend",
        messages=[ChatMessage(content="Summarize the incident and next steps.")],
        max_tokens=max_tokens,
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for a domain."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _strategy(
    *,
    health: CircuitBreakerRegistry | None = None,
    success_stats: SuccessStats | None = None,
    latency_stats: LatencyStats | None = None,
    success_weight: float = 0.35,
    latency_weight: float = 0.25,
    quality_weight: float = 0.25,
    cost_weight: float = 0.15,
    catalog: dict[str, ModelCandidate] | None = None,
) -> ProviderHealthScoreBlendStrategy:
    """Build a provider-health blend strategy with overridable dependencies."""
    return ProviderHealthScoreBlendStrategy(
        catalog or default_model_catalog(),
        health or CircuitBreakerRegistry(),
        success_stats or SuccessStats(),
        latency_stats or LatencyStats(),
        success_weight=success_weight,
        latency_weight=latency_weight,
        quality_weight=quality_weight,
        cost_weight=cost_weight,
    )


def test_health_blend_success_weight_prefers_reliable_provider() -> None:
    """A degraded top-quality provider loses when success rate dominates."""
    stats = SuccessStats()
    stats.observe("anthropic", success=False)
    stats.observe("openai", success=True)
    stats.observe("openai", success=True)
    strategy = _strategy(
        success_stats=stats,
        success_weight=1.0,
        latency_weight=0.0,
        quality_weight=0.0,
        cost_weight=0.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_HEALTH_SCORE_BLEND
    assert "success=100.00%" in decision.rationale


def test_health_blend_skips_open_circuit_when_any_provider_is_healthy() -> None:
    """Open circuits are hard-gated out of primary scoring when possible."""
    health = CircuitBreakerRegistry(failure_threshold=1, recovery_window_seconds=60.0)
    health.record_failure("anthropic")
    strategy = _strategy(
        health=health,
        success_weight=0.0,
        latency_weight=0.0,
        quality_weight=1.0,
        cost_weight=0.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.chosen_model != ANTHROPIC_SAFETY_MODEL
    assert decision.fallback_chain[0] != ANTHROPIC_SAFETY_MODEL
    assert "open circuits excluded" in decision.rationale


def test_health_blend_scores_all_candidates_when_every_circuit_is_open() -> None:
    """When all circuits are open, decide-time still returns the top score."""
    health = CircuitBreakerRegistry(failure_threshold=1, recovery_window_seconds=60.0)
    for provider in ("openai", "anthropic", "google", "moonshot"):
        health.record_failure(provider)
    strategy = _strategy(
        health=health,
        success_weight=0.0,
        latency_weight=0.0,
        quality_weight=1.0,
        cost_weight=0.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "no closed circuits" in decision.rationale


def test_health_blend_empty_latency_and_equal_costs_do_not_divide_by_zero() -> None:
    """Equal cost/latency normalization should be neutral and safe."""
    catalog = {
        "steady-a": ModelCandidate(
            model="steady-a",
            provider="steady-a",
            quality_score=0.70,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.001,
            supports_domains={DomainTag.GENERAL},
        ),
        "steady-b": ModelCandidate(
            model="steady-b",
            provider="steady-b",
            quality_score=0.90,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.001,
            supports_domains={DomainTag.GENERAL},
        ),
    }
    strategy = _strategy(
        catalog=catalog,
        success_weight=0.0,
        latency_weight=0.5,
        quality_weight=0.25,
        cost_weight=0.25,
    )

    decision = strategy.choose(_request(max_tokens=128), _signals())

    assert decision.chosen_model == "steady-b"
    assert "p95=0.0ms" in decision.rationale


def test_health_blend_zero_weights_fall_back_to_quality() -> None:
    """All-zero weights degrade to pure quality instead of failing."""
    strategy = _strategy(
        success_weight=0.0,
        latency_weight=0.0,
        quality_weight=0.0,
        cost_weight=0.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_health_blend_rejects_negative_weight() -> None:
    """Negative weights should fail fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        _strategy(success_weight=-0.1)


def test_health_blend_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose provider-health-score-blend."""
    catalog = default_model_catalog()
    settings = RouterSettings(
        health_blend_success_weight=0.4,
        health_blend_latency_weight=0.2,
        health_blend_quality_weight=0.3,
        health_blend_cost_weight=0.1,
    )
    strategies = build_strategies(
        catalog,
        LatencyStats(),
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
    )

    assert (
        strategies[RoutingStrategyName.PROVIDER_HEALTH_SCORE_BLEND].strategy_name
        is RoutingStrategyName.PROVIDER_HEALTH_SCORE_BLEND
    )
