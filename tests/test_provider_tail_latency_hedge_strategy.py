"""Tests for provider-tail-latency-hedge routing."""

import pytest

from router.config import RouterSettings
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
    InflightStats,
    LatencyStats,
    ProviderTailLatencyHedgeStrategy,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _catalog() -> dict[str, ModelCandidate]:
    return {
        "quality-primary": ModelCandidate(
            model="quality-primary",
            provider="openai",
            quality_score=0.95,
            input_cost_per_1k=0.006,
            output_cost_per_1k=0.018,
            supports_domains={DomainTag.GENERAL},
        ),
        "fast-alternate": ModelCandidate(
            model="fast-alternate",
            provider="google",
            quality_score=0.85,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.01,
            supports_domains={DomainTag.GENERAL},
        ),
    }


def _request() -> RouterRequest:
    return RouterRequest(
        request_id="req-tail-latency",
        messages=[ChatMessage(content="Route around a tail latency spike")],
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=100,
    )


def _strategy(
    latency_stats: LatencyStats,
    *,
    unavailable: set[str] | None = None,
    threshold_ms: float = 1000.0,
) -> ProviderTailLatencyHedgeStrategy:
    return ProviderTailLatencyHedgeStrategy(
        _catalog(),
        latency_stats,
        _FakeHealth(unavailable),
        tail_latency_threshold_ms=threshold_ms,
    )


def test_provider_tail_latency_hedge_enum_parses() -> None:
    assert (
        RoutingStrategyName("provider-tail-latency-hedge")
        is RoutingStrategyName.PROVIDER_TAIL_LATENCY_HEDGE
    )


def test_provider_tail_latency_hedge_cold_start_keeps_quality_leader() -> None:
    decision = _strategy(LatencyStats()).choose(_request(), _signals())

    assert decision.chosen_model == "quality-primary"
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_TAIL_LATENCY_HEDGE
    assert "no observed healthy provider alternative" in decision.rationale


def test_provider_tail_latency_hedge_uses_p95_not_median() -> None:
    stats = LatencyStats()
    for latency_ms in [100.0] * 18 + [1600.0, 1700.0]:
        stats.observe("openai", latency_ms)
    for _ in range(20):
        stats.observe("google", 250.0)

    decision = _strategy(stats).choose(_request(), _signals())

    assert stats.p50("openai") == 100.0
    assert stats.p95("openai") == 1600.0
    assert decision.chosen_model == "fast-alternate"
    assert "hedged across providers" in decision.rationale


def test_provider_tail_latency_hedge_does_not_trigger_under_p95_threshold() -> None:
    stats = LatencyStats()
    for latency_ms in [100.0] * 18 + [800.0, 900.0]:
        stats.observe("openai", latency_ms)
    for _ in range(20):
        stats.observe("google", 200.0)

    decision = _strategy(stats).choose(_request(), _signals())

    assert decision.chosen_model == "quality-primary"
    assert "tail p95 remained within threshold" in decision.rationale


def test_provider_tail_latency_hedge_ignores_unavailable_alternative() -> None:
    stats = LatencyStats()
    for _ in range(20):
        stats.observe("openai", 1800.0)
        stats.observe("google", 200.0)

    decision = _strategy(stats, unavailable={"google"}).choose(_request(), _signals())

    assert decision.chosen_model == "quality-primary"
    assert "no observed healthy provider alternative" in decision.rationale


def test_provider_tail_latency_hedge_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        _strategy(LatencyStats(), threshold_ms=-1.0)


def test_provider_tail_latency_hedge_registered_by_strategy_factory() -> None:
    settings = RouterSettings(provider_tail_latency_hedge_ms=1200.0)
    catalog = _catalog()
    latency_stats = LatencyStats()
    strategies = build_strategies(
        catalog,
        latency_stats,
        InflightStats(),
        settings.quality_floor,
        "quality-primary",
        "fast-alternate",
        settings.ab_model_a_weight,
        CircuitBreakerRegistry(),
        settings.blend_quality_weight,
        settings.blend_cost_weight,
        settings.blend_latency_weight,
        settings.request_cost_ceiling_usd,
        "quality-primary",
        "fast-alternate",
        settings.canary_weight,
        settings.latency_sla_ms,
        provider_tail_latency_hedge_ms=settings.provider_tail_latency_hedge_ms,
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_TAIL_LATENCY_HEDGE]
    assert isinstance(strategy, ProviderTailLatencyHedgeStrategy)
    assert strategy._latency_stats is latency_stats  # noqa: SLF001
    assert strategy._tail_latency_threshold_ms == 1200.0  # noqa: SLF001
    assert RouterSettings().provider_tail_latency_hedge_ms == 1500.0
