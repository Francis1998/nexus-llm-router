"""Tests for the latency-slope-shed routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    OPENAI_BALANCED_MODEL,
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
    LatencySlopeShedStrategy,
    LatencySlopeStats,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _AlwaysHealthy:
    """Provider health stub that treats every provider as available."""

    def is_available(self, provider: str) -> bool:
        return True


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(request_id="req-slope-shed", messages=[ChatMessage(content="hello")])


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_latency_slope_shed_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("latency-slope-shed") is RoutingStrategyName.LATENCY_SLOPE_SHED


def test_latency_slope_shed_cold_start_keeps_top_quality() -> None:
    """With fewer than two samples the EWMA slope is flat, so quality wins."""
    strategy = LatencySlopeShedStrategy(
        default_model_catalog(),
        LatencySlopeStats(window=10),
        _AlwaysHealthy(),
        latency_slope_threshold_ms=25.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.LATENCY_SLOPE_SHED
    assert "kept highest quality" in decision.rationale


def test_latency_slope_shed_sheds_rising_quality_leader() -> None:
    """A rising anthropic EWMA slope sheds to a lower-latency / cheaper model.

    Claude Sonnet 4.6 leads catalog quality; climbing anthropic latency must
    fall through toward GPT-5.5 / cheaper arms instead of staying on Claude.
    """
    stats = LatencySlopeStats(window=6, alpha=0.5)
    for latency in (100.0, 200.0, 350.0, 550.0, 800.0, 1200.0):
        stats.observe("anthropic", latency)
    for latency in (80.0, 85.0, 90.0, 88.0, 92.0, 91.0):
        stats.observe("openai", latency)
        stats.observe("google", latency)
        stats.observe("moonshot", latency)
    strategy = LatencySlopeShedStrategy(
        default_model_catalog(),
        stats,
        _AlwaysHealthy(),
        latency_slope_threshold_ms=25.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model != ANTHROPIC_SAFETY_MODEL
    assert "shed rising" in decision.rationale
    assert decision.provider in {"openai", "google", "moonshot"}


def test_latency_slope_shed_prefers_cheaper_when_means_tie() -> None:
    """Among shed targets, lower mean latency then lower cost wins."""
    stats = LatencySlopeStats(window=5, alpha=0.5)
    for latency in (50.0, 150.0, 300.0, 500.0, 900.0):
        stats.observe("anthropic", latency)
    for _ in range(5):
        stats.observe("openai", 100.0)
        stats.observe("google", 100.0)
        stats.observe("moonshot", 100.0)
    strategy = LatencySlopeShedStrategy(
        default_model_catalog(),
        stats,
        _AlwaysHealthy(),
        latency_slope_threshold_ms=10.0,
    )

    decision = strategy.choose(_request(), _signals())

    # openai balanced is cheapest among equal-mean shed pool
    assert decision.chosen_model == OPENAI_BALANCED_MODEL


def test_latency_slope_shed_respects_domain_support() -> None:
    """Only medical-capable models are considered for a medical prompt."""
    strategy = LatencySlopeShedStrategy(
        default_model_catalog(),
        LatencySlopeStats(),
        _AlwaysHealthy(),
        latency_slope_threshold_ms=25.0,
    )

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    candidate = default_model_catalog()[decision.chosen_model]
    assert DomainTag.MEDICAL in candidate.supports_domains
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_latency_slope_shed_rejects_negative_threshold() -> None:
    """A negative slope threshold fails fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        LatencySlopeShedStrategy(
            default_model_catalog(),
            LatencySlopeStats(),
            _AlwaysHealthy(),
            latency_slope_threshold_ms=-1.0,
        )


def test_latency_slope_stats_rejects_small_window() -> None:
    """A window smaller than two samples fails fast."""
    with pytest.raises(ValueError, match="window"):
        LatencySlopeStats(window=1)


def test_latency_slope_shed_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose latency-slope-shed."""
    catalog = default_model_catalog()
    settings = RouterSettings(latency_slope_window=8, latency_slope_threshold_ms=15.0)
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
        latency_slope_window=settings.latency_slope_window,
        latency_slope_threshold_ms=settings.latency_slope_threshold_ms,
    )

    strategy = strategies[RoutingStrategyName.LATENCY_SLOPE_SHED]
    assert isinstance(strategy, LatencySlopeShedStrategy)
    assert strategy.strategy_name is RoutingStrategyName.LATENCY_SLOPE_SHED


def test_latency_slope_shed_keeps_frontier_when_slope_flat() -> None:
    """Flat openai slope with rising others still allows GPT-5.5 quality path.

    Sanity check that non-rising providers are not shed: with anthropic flat
    and under threshold, Claude Sonnet 4.6 remains selected.
    """
    stats = LatencySlopeStats(window=4, alpha=0.5)
    for latency in (200.0, 205.0, 198.0, 202.0):
        stats.observe("anthropic", latency)
    strategy = LatencySlopeShedStrategy(
        default_model_catalog(),
        stats,
        _AlwaysHealthy(),
        latency_slope_threshold_ms=25.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
