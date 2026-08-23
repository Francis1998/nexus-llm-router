"""Tests for provider-warmup-weight routing."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, MOONSHOT_BALANCED_MODEL
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
    ProviderWarmupWeightStrategy,
    SuccessStats,
    WarmupStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _request(metadata: dict | None = None) -> RouterRequest:
    return RouterRequest(
        request_id="req-warmup-weight",
        messages=[ChatMessage(content="Route with a provider warmup weight.")],
        metadata=metadata or {},
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
    stats: WarmupStats | None = None,
    *,
    blend: float = 0.3,
    unavailable: set[str] | None = None,
) -> ProviderWarmupWeightStrategy:
    return ProviderWarmupWeightStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        stats or WarmupStats(),
        warmup_blend=blend,
    )


def test_provider_warmup_weight_enum_parses() -> None:
    assert (
        RoutingStrategyName("provider-warmup-weight") is RoutingStrategyName.PROVIDER_WARMUP_WEIGHT
    )


def test_provider_warmup_weight_rejects_invalid_blend() -> None:
    with pytest.raises(ValueError, match="warmup_blend must be within"):
        _strategy(blend=1.5)


def test_warmup_stats_rejects_invalid_lookback() -> None:
    with pytest.raises(ValueError, match="lookback must be >= 1"):
        WarmupStats(lookback=0)


def test_warmup_stats_cold_provider_scores_zero() -> None:
    stats = WarmupStats()
    assert stats.warmup_score("openai") == 0.0


def test_warmup_stats_score_reflects_recent_successes() -> None:
    stats = WarmupStats(lookback=10)
    for _ in range(5):
        stats.observe("openai", success=True)
    assert stats.warmup_score("openai") == pytest.approx(0.5)


def test_warmup_stats_evicts_beyond_lookback() -> None:
    stats = WarmupStats(lookback=4)
    for _ in range(10):
        stats.observe("openai", success=True)
    assert stats.warmup_score("openai") == pytest.approx(1.0)


def test_provider_warmup_weight_cold_start_is_quality_first() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_WARMUP_WEIGHT
    assert "warmup 0.00" in decision.rationale


def test_provider_warmup_weight_biases_toward_warm_provider() -> None:
    stats = WarmupStats()
    for _ in range(50):
        stats.observe("moonshot", success=True)
    decision = _strategy(stats, blend=0.5).choose(_request(), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert decision.provider == "moonshot"


def test_provider_warmup_weight_respects_metadata_override() -> None:
    decision = _strategy(blend=0.3).choose(
        _request({"provider_warmup_score": {"moonshot": 1.0}}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_provider_warmup_weight_clamps_out_of_range_override() -> None:
    strategy = _strategy(blend=1.0)
    decision = strategy.choose(_request({"provider_warmup_score": {"moonshot": 5.0}}), _signals())

    assert "warmup 1.00" in decision.rationale
    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_provider_warmup_weight_ignores_malformed_override() -> None:
    strategy = _strategy(blend=0.5)
    decision = strategy.choose(
        _request({"provider_warmup_score": {"moonshot": "not-a-number"}}), _signals()
    )

    assert "warmup 0.00" in decision.rationale


def test_provider_warmup_weight_respects_circuit_health() -> None:
    stats = WarmupStats()
    for _ in range(50):
        stats.observe("anthropic", success=True)
    decision = _strategy(stats, blend=0.5, unavailable={"anthropic"}).choose(_request(), _signals())

    assert decision.provider != "anthropic"


def test_provider_warmup_weight_registered_by_strategy_factory() -> None:
    settings = RouterSettings(provider_warmup_blend=0.4)
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
        success_stats=SuccessStats(),
        provider_warmup_blend=settings.provider_warmup_blend,
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_WARMUP_WEIGHT]
    assert isinstance(strategy, ProviderWarmupWeightStrategy)
    assert strategy._warmup_blend == 0.4  # noqa: SLF001
    assert RouterSettings().provider_warmup_blend == 0.3
