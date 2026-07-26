"""Tests for the adaptive-timeout routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    MOONSHOT_BALANCED_MODEL,
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
    AdaptiveTimeoutStrategy,
    InflightStats,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(
        request_id="req-adaptive-timeout",
        messages=[ChatMessage(content="Summarize the incident and recommend next steps.")],
    )


def _signals(
    domain_tag: DomainTag = DomainTag.GENERAL,
    latency_requirement: LatencyRequirement = LatencyRequirement.REALTIME,
) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=latency_requirement,
        token_budget=4096,
        prompt_tokens_estimate=96,
    )


def _strategy(
    *,
    latency_stats: LatencyStats | None = None,
    success_stats: SuccessStats | None = None,
    base_timeout_ms: float = 750.0,
) -> AdaptiveTimeoutStrategy:
    """Build an adaptive-timeout strategy with overridable signal stores."""
    return AdaptiveTimeoutStrategy(
        default_model_catalog(),
        latency_stats or LatencyStats(),
        success_stats or SuccessStats(),
        base_timeout_ms,
    )


def test_adaptive_timeout_cold_start_picks_top_quality_model() -> None:
    """With no latency or error observations, every provider fits the budget."""
    strategy = _strategy()

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.ADAPTIVE_TIMEOUT
    assert "adaptive-timeout" in decision.rationale
    assert ANTHROPIC_SAFETY_MODEL not in decision.fallback_chain


def test_adaptive_timeout_realtime_pressure_prefers_fast_model() -> None:
    """A tight realtime budget trades quality for providers with lower recent p95."""
    latency_stats = LatencyStats()
    latency_stats.observe("anthropic", 1100.0)
    latency_stats.observe("openai", 900.0)
    latency_stats.observe("google", 180.0)
    latency_stats.observe("moonshot", 450.0)
    strategy = _strategy(latency_stats=latency_stats)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == GEMINI_FLASH_MODEL
    assert "adaptive timeout budget" in decision.rationale


def test_adaptive_timeout_batch_budget_allows_slower_quality_model() -> None:
    """Batch requests keep more quality headroom when observed p95 is comfortable."""
    latency_stats = LatencyStats()
    latency_stats.observe("anthropic", 2200.0)
    latency_stats.observe("openai", 900.0)
    latency_stats.observe("google", 400.0)
    latency_stats.observe("moonshot", 600.0)
    strategy = _strategy(latency_stats=latency_stats)

    decision = strategy.choose(_request(), _signals(latency_requirement=LatencyRequirement.BATCH))

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "provider p95 2200.0ms" in decision.rationale


def test_adaptive_timeout_penalizes_recent_provider_errors() -> None:
    """Recent failures inflate risk-adjusted latency and can skip a quality leader."""
    latency_stats = LatencyStats()
    latency_stats.observe("anthropic", 400.0)
    latency_stats.observe("openai", 450.0)
    latency_stats.observe("google", 250.0)
    latency_stats.observe("moonshot", 350.0)
    success_stats = SuccessStats()
    success_stats.observe("anthropic", success=True)
    success_stats.observe("anthropic", success=False)
    success_stats.observe("openai", success=True)
    success_stats.observe("openai", success=True)
    strategy = _strategy(latency_stats=latency_stats, success_stats=success_stats)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.chosen_model != ANTHROPIC_SAFETY_MODEL
    assert "success 100.00%" in decision.rationale


def test_adaptive_timeout_unreachable_budget_falls_back_to_lowest_risk_latency() -> None:
    """When no provider fits the adaptive budget, route to lowest risk-adjusted latency."""
    latency_stats = LatencyStats()
    latency_stats.observe("anthropic", 1500.0)
    latency_stats.observe("openai", 1200.0)
    latency_stats.observe("google", 1000.0)
    latency_stats.observe("moonshot", 900.0)
    strategy = _strategy(latency_stats=latency_stats)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "no provider within" in decision.rationale


def test_adaptive_timeout_respects_domain_support() -> None:
    """Only domain-eligible candidates are considered for specialized requests."""
    strategy = _strategy()

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    candidate = default_model_catalog()[decision.chosen_model]
    assert DomainTag.MEDICAL in candidate.supports_domains
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_adaptive_timeout_rejects_negative_base_timeout() -> None:
    """A negative timeout base fails fast at construction."""
    with pytest.raises(ValueError, match="base_timeout_ms must be non-negative"):
        _strategy(base_timeout_ms=-1.0)


def test_adaptive_timeout_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose adaptive-timeout."""
    settings = RouterSettings()
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
        strategies[RoutingStrategyName.ADAPTIVE_TIMEOUT].strategy_name
        is RoutingStrategyName.ADAPTIVE_TIMEOUT
    )
