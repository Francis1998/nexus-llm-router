"""Tests for the provider-weight-decay routing strategy."""

from __future__ import annotations

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_FRONTIER_MODEL
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
    ProviderWeightDecayStrategy,
    ProviderWeightStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id="req-provider-weight-decay",
        messages=[ChatMessage(content="Decay failing providers.")],
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _strategy(stats: ProviderWeightStats | None = None) -> ProviderWeightDecayStrategy:
    """Build provider-weight-decay with overridable stats."""
    return ProviderWeightDecayStrategy(
        default_model_catalog(),
        stats or ProviderWeightStats(0.5, 0.1),
    )


def test_provider_weight_decay_enum_parses() -> None:
    """The API header parser can resolve the strategy value."""
    assert RoutingStrategyName("provider-weight-decay") is RoutingStrategyName.PROVIDER_WEIGHT_DECAY


def test_provider_weight_stats_rejects_invalid_decay() -> None:
    """Decay factor must stay within (0, 1]."""
    with pytest.raises(ValueError, match="decay_factor"):
        ProviderWeightStats(0.0, 0.1)


def test_provider_weight_stats_rejects_negative_recover() -> None:
    """Recovery step must be non-negative."""
    with pytest.raises(ValueError, match="recover"):
        ProviderWeightStats(0.5, -0.1)


def test_provider_weight_decay_cold_start_picks_top_quality() -> None:
    """Unseen providers start at weight 1.0 and keep quality-first ranking."""
    decision = _strategy().choose(_request(), _signals())
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_WEIGHT_DECAY
    assert "weight 1.000" in decision.rationale


def test_provider_weight_decay_shifts_after_failures() -> None:
    """Repeated failures decay the top provider below the next quality candidate."""
    stats = ProviderWeightStats(0.5, 0.1)
    stats.observe("anthropic", success=False)
    stats.observe("anthropic", success=False)
    assert stats.weight("anthropic") == pytest.approx(0.25)
    decision = _strategy(stats).choose(_request(), _signals())
    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "weight 1.000" in decision.rationale


def test_provider_weight_decay_recovers_slowly_on_success() -> None:
    """Success adds the recover step and caps weight at 1.0."""
    stats = ProviderWeightStats(0.5, 0.1)
    stats.observe("openai", success=False)
    assert stats.weight("openai") == pytest.approx(0.5)
    stats.observe("openai", success=True)
    assert stats.weight("openai") == pytest.approx(0.6)
    stats.observe("openai", success=True)
    stats.observe("openai", success=True)
    stats.observe("openai", success=True)
    stats.observe("openai", success=True)
    assert stats.weight("openai") == pytest.approx(1.0)


def test_provider_weight_decay_registered_by_strategy_factory() -> None:
    """The built-in strategy map exposes provider-weight-decay."""
    settings = RouterSettings(
        provider_weight_decay_factor=0.4,
        provider_weight_recover=0.2,
    )
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
        provider_weight_decay_factor=settings.provider_weight_decay_factor,
        provider_weight_recover=settings.provider_weight_recover,
    )
    strategy = strategies[RoutingStrategyName.PROVIDER_WEIGHT_DECAY]
    assert isinstance(strategy, ProviderWeightDecayStrategy)
    assert strategy._provider_weight_stats.decay_factor == 0.4  # noqa: SLF001
    assert strategy._provider_weight_stats.recover == 0.2  # noqa: SLF001


def test_provider_weight_decay_settings_defaults() -> None:
    """RouterSettings expose decay and recover defaults."""
    settings = RouterSettings()
    assert settings.provider_weight_decay_factor == 0.5
    assert settings.provider_weight_recover == 0.1
