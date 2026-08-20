"""Tests for provider-cold-start-bias routing."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_PRO_MODEL,
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
    LatencyStats,
    ProviderColdStartBiasStrategy,
    ProviderColdStartStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _request() -> RouterRequest:
    return RouterRequest(
        request_id="req-provider-cold-start",
        messages=[ChatMessage(content="Explore provider coverage safely.")],
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _strategy(
    stats: ProviderColdStartStats | None = None,
    *,
    unavailable: set[str] | None = None,
    target: int = 5,
) -> ProviderColdStartBiasStrategy:
    return ProviderColdStartBiasStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        stats or ProviderColdStartStats(),
        observation_target=target,
    )


def test_provider_cold_start_bias_enum_parses() -> None:
    assert (
        RoutingStrategyName("provider-cold-start-bias")
        is RoutingStrategyName.PROVIDER_COLD_START_BIAS
    )


def test_provider_cold_start_stats_rejects_invalid_lookback() -> None:
    with pytest.raises(ValueError, match="lookback must be >= 1"):
        ProviderColdStartStats(0)


def test_provider_cold_start_stats_evicts_stale_observations() -> None:
    stats = ProviderColdStartStats(lookback=2)
    stats.observe("anthropic")
    stats.observe("openai")
    stats.observe("openai")

    assert stats.observation_count("anthropic") == 0
    assert stats.observation_count("openai") == 2


def test_provider_cold_start_bias_cold_start_uses_quality_tie_break() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_COLD_START_BIAS
    assert "cold start" in decision.rationale
    assert "0/5 observations" in decision.rationale


def test_provider_cold_start_bias_prefers_least_observed_provider() -> None:
    stats = ProviderColdStartStats()
    for provider in ("anthropic", "openai", "moonshot"):
        stats.observe(provider)

    decision = _strategy(stats).choose(_request(), _signals())

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert decision.provider == "google"
    assert "least-observed healthy provider google at 0/5" in decision.rationale


def test_provider_cold_start_bias_skips_unhealthy_exploration_gap() -> None:
    stats = ProviderColdStartStats()
    for provider in ("anthropic", "openai", "moonshot"):
        stats.observe(provider)

    decision = _strategy(stats, unavailable={"google"}).choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.provider != "google"


def test_provider_cold_start_bias_returns_to_quality_when_coverage_is_warm() -> None:
    stats = ProviderColdStartStats()
    for provider in ("anthropic", "openai", "google", "moonshot"):
        stats.observe(provider)

    decision = _strategy(stats, target=1).choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "coverage warm" in decision.rationale


def test_provider_cold_start_bias_rejects_invalid_target() -> None:
    with pytest.raises(ValueError, match="observation_target must be >= 1"):
        _strategy(target=0)


def test_provider_cold_start_bias_registered_by_strategy_factory() -> None:
    settings = RouterSettings(
        provider_cold_start_lookback=17,
        provider_cold_start_target=3,
    )
    stats = ProviderColdStartStats(settings.provider_cold_start_lookback)
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
        provider_cold_start_stats=stats,
        provider_cold_start_lookback=settings.provider_cold_start_lookback,
        provider_cold_start_target=settings.provider_cold_start_target,
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_COLD_START_BIAS]
    assert isinstance(strategy, ProviderColdStartBiasStrategy)
    assert strategy._observation_stats is stats  # noqa: SLF001
    assert strategy._observation_target == 3  # noqa: SLF001
    assert RouterSettings().provider_cold_start_lookback == 100
    assert RouterSettings().provider_cold_start_target == 5
