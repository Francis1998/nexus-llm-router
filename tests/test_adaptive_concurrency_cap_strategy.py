"""Tests for the adaptive-concurrency-cap routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    AdaptiveConcurrencyCapStrategy,
    InflightStats,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    return RouterRequest(request_id="req-adaptive-cap", messages=[ChatMessage(content="hello")])


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=100,
    )


def _strategy(
    *,
    base_cap: int = 8,
    min_cap: int = 1,
    latency_reference_ms: float = 2000.0,
) -> AdaptiveConcurrencyCapStrategy:
    return AdaptiveConcurrencyCapStrategy(
        default_model_catalog(),
        InflightStats(),
        SuccessStats(),
        LatencyStats(),
        base_cap=base_cap,
        min_cap=min_cap,
        latency_reference_ms=latency_reference_ms,
    )


def test_adaptive_concurrency_cap_enum_parses() -> None:
    assert (
        RoutingStrategyName("adaptive-concurrency-cap")
        is RoutingStrategyName.ADAPTIVE_CONCURRENCY_CAP
    )


def test_adaptive_concurrency_cap_cold_start_prefers_quality_leader() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.ADAPTIVE_CONCURRENCY_CAP
    assert "adaptive-concurrency-cap selected below adaptive cap" in decision.rationale


def test_adaptive_concurrency_cap_tightens_cap_for_errors() -> None:
    inflight = InflightStats()
    success = SuccessStats()
    for _ in range(4):
        success.observe("anthropic", success=False)
    for _ in range(4):
        success.observe("anthropic", success=True)
    for _ in range(3):
        inflight.begin("anthropic")
    strategy = AdaptiveConcurrencyCapStrategy(
        default_model_catalog(),
        inflight,
        success,
        LatencyStats(),
        base_cap=8,
        min_cap=1,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.provider == "openai"
    assert "adaptive cap" in decision.rationale


def test_adaptive_concurrency_cap_tightens_cap_for_latency() -> None:
    inflight = InflightStats()
    latency = LatencyStats()
    for _ in range(10):
        latency.observe("anthropic", 4000.0)
    for _ in range(4):
        inflight.begin("anthropic")
    strategy = AdaptiveConcurrencyCapStrategy(
        default_model_catalog(),
        inflight,
        SuccessStats(),
        latency,
        base_cap=8,
        min_cap=1,
        latency_reference_ms=2000.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.provider == "openai"
    assert "adaptive cap" in decision.rationale


def test_adaptive_concurrency_cap_falls_back_when_all_saturated() -> None:
    inflight = InflightStats()
    for provider in ("anthropic", "openai", "google", "moonshot"):
        for _ in range(8):
            inflight.begin(provider)
    strategy = AdaptiveConcurrencyCapStrategy(
        default_model_catalog(),
        inflight,
        SuccessStats(),
        LatencyStats(),
        base_cap=8,
    )

    decision = strategy.choose(_request(), _signals())

    assert "least-saturated fallback" in decision.rationale


def test_adaptive_concurrency_cap_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="min_cap"):
        AdaptiveConcurrencyCapStrategy(
            default_model_catalog(),
            InflightStats(),
            SuccessStats(),
            LatencyStats(),
            base_cap=4,
            min_cap=8,
        )


def test_adaptive_concurrency_cap_registered_by_strategy_factory() -> None:
    settings = RouterSettings(
        adaptive_concurrency_base_cap=6,
        adaptive_concurrency_min_cap=2,
        adaptive_concurrency_latency_ms=1500.0,
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
        adaptive_concurrency_base_cap=settings.adaptive_concurrency_base_cap,
        adaptive_concurrency_min_cap=settings.adaptive_concurrency_min_cap,
        adaptive_concurrency_latency_ms=settings.adaptive_concurrency_latency_ms,
    )

    strategy = strategies[RoutingStrategyName.ADAPTIVE_CONCURRENCY_CAP]
    assert isinstance(strategy, AdaptiveConcurrencyCapStrategy)
    assert strategy._base_cap == 6  # noqa: SLF001
    assert strategy._min_cap == 2  # noqa: SLF001
    assert RouterSettings().adaptive_concurrency_base_cap == 8
