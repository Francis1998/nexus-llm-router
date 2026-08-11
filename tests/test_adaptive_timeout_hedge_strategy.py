"""Tests for the adaptive-timeout-hedge routing strategy."""

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
    AdaptiveTimeoutHedgeStrategy,
    InflightStats,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _catalog() -> dict[str, ModelCandidate]:
    """Build a catalog with quality and latency trade-offs."""
    return {
        "quality-leader": ModelCandidate(
            model="quality-leader",
            provider="anthropic",
            quality_score=0.99,
            input_cost_per_1k=0.006,
            output_cost_per_1k=0.018,
            supports_domains={DomainTag.GENERAL},
        ),
        "balanced": ModelCandidate(
            model="balanced",
            provider="openai",
            quality_score=0.90,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.010,
            supports_domains={DomainTag.GENERAL},
        ),
        "fast": ModelCandidate(
            model="fast",
            provider="moonshot",
            quality_score=0.76,
            input_cost_per_1k=0.0005,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
        ),
    }


def _request() -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id="req-adaptive-timeout-hedge",
        messages=[ChatMessage(content="Choose a provider.")],
    )


def _signals() -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def test_adaptive_timeout_hedge_enum_parses() -> None:
    """The API header parser can resolve the strategy value."""
    assert (
        RoutingStrategyName("adaptive-timeout-hedge") is RoutingStrategyName.ADAPTIVE_TIMEOUT_HEDGE
    )


def test_adaptive_timeout_hedge_rejects_ratio_below_one() -> None:
    """A sub-unity ratio would make the fastest provider exceed its own threshold."""
    with pytest.raises(ValueError, match="hedge_ratio must be >= 1.0"):
        AdaptiveTimeoutHedgeStrategy(_catalog(), LatencyStats(), hedge_ratio=0.99)


def test_adaptive_timeout_hedge_cold_start_keeps_quality_leader() -> None:
    """Unknown provider latency should not trigger a speculative hedge."""
    strategy = AdaptiveTimeoutHedgeStrategy(_catalog(), LatencyStats())

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == "quality-leader"
    assert decision.routing_strategy is RoutingStrategyName.ADAPTIVE_TIMEOUT_HEDGE
    assert "insufficient positive p95 observations" in decision.rationale


def test_adaptive_timeout_hedge_selects_fastest_when_quality_leader_is_slow() -> None:
    """A quality leader above the adaptive threshold should hedge to the fastest peer."""
    latency_stats = LatencyStats()
    latency_stats.observe("anthropic", 900.0)
    latency_stats.observe("openai", 400.0)
    latency_stats.observe("moonshot", 200.0)
    strategy = AdaptiveTimeoutHedgeStrategy(_catalog(), latency_stats, hedge_ratio=1.5)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == "fast"
    assert "hedged from quality leader" in decision.rationale
    assert "adaptive threshold 300.0ms" in decision.rationale


def test_adaptive_timeout_hedge_stays_when_leader_is_within_threshold() -> None:
    """Small latency differences should not displace the quality leader."""
    latency_stats = LatencyStats()
    latency_stats.observe("anthropic", 500.0)
    latency_stats.observe("openai", 450.0)
    latency_stats.observe("moonshot", 400.0)
    strategy = AdaptiveTimeoutHedgeStrategy(_catalog(), latency_stats, hedge_ratio=1.5)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == "quality-leader"
    assert "within adaptive threshold 600.0ms" in decision.rationale


def test_adaptive_timeout_hedge_does_not_treat_unknown_latency_as_fast() -> None:
    """An unobserved alternative with p95 zero must not attract a hedge."""
    latency_stats = LatencyStats()
    latency_stats.observe("anthropic", 900.0)
    strategy = AdaptiveTimeoutHedgeStrategy(_catalog(), latency_stats, hedge_ratio=1.5)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == "quality-leader"
    assert "no faster observed provider alternative" in decision.rationale


def test_adaptive_timeout_hedge_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose adaptive-timeout-hedge."""
    settings = RouterSettings(adaptive_timeout_hedge_ratio=2.0)
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
        adaptive_timeout_hedge_ratio=settings.adaptive_timeout_hedge_ratio,
    )

    strategy = strategies[RoutingStrategyName.ADAPTIVE_TIMEOUT_HEDGE]
    assert isinstance(strategy, AdaptiveTimeoutHedgeStrategy)


def test_adaptive_timeout_hedge_settings_default() -> None:
    """RouterSettings expose a conservative adaptive hedge ratio."""
    assert RouterSettings().adaptive_timeout_hedge_ratio == 1.5
