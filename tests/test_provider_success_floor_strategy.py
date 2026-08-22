"""Tests for provider-success-floor routing."""

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
    InflightStats,
    LatencyStats,
    ProviderSuccessFloorStrategy,
    SuccessStats,
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
        request_id="req-success-floor",
        messages=[ChatMessage(content="Route with a provider success floor.")],
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _success_stats() -> SuccessStats:
    stats = SuccessStats()
    for _ in range(10):
        stats.observe("anthropic", success=True)
        stats.observe("google", success=True)
        stats.observe("moonshot", success=True)
        stats.observe("openai", success=True)
    for _ in range(4):
        stats.observe("anthropic", success=False)
    for _ in range(1):
        stats.observe("google", success=False)
    return stats


def _strategy(
    stats: SuccessStats | None = None,
    *,
    floor: float = 0.85,
    unavailable: set[str] | None = None,
) -> ProviderSuccessFloorStrategy:
    return ProviderSuccessFloorStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        stats or _success_stats(),
        success_floor=floor,
    )


def test_provider_success_floor_enum_parses() -> None:
    assert (
        RoutingStrategyName("provider-success-floor") is RoutingStrategyName.PROVIDER_SUCCESS_FLOOR
    )


def test_provider_success_floor_rejects_invalid_floor() -> None:
    with pytest.raises(ValueError, match="success_floor must be within"):
        _strategy(floor=1.5)


def test_provider_success_floor_skips_providers_below_floor() -> None:
    decision = _strategy().choose(_request(), _signals())

    # anthropic is ~10/14 ~= 0.714 below 0.85; google is ~10/11 ~= 0.909
    assert decision.chosen_model != ANTHROPIC_SAFETY_MODEL
    assert "meets floor" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_SUCCESS_FLOOR


def test_provider_success_floor_prefers_quality_among_eligible() -> None:
    stats = SuccessStats()
    for provider in ("anthropic", "google", "moonshot", "openai"):
        for _ in range(20):
            stats.observe(provider, success=True)
    decision = _strategy(stats).choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "highest-quality provider meeting success floor" in decision.rationale


def test_provider_success_floor_emergency_retains_when_all_below() -> None:
    stats = SuccessStats()
    for provider, failures in (("anthropic", 5), ("google", 8), ("moonshot", 6), ("openai", 7)):
        for _ in range(10):
            stats.observe(provider, success=True)
        for _ in range(failures):
            stats.observe(provider, success=False)

    decision = _strategy(stats, floor=0.95).choose(_request(), _signals())

    assert "emergency" in decision.rationale.lower()
    assert "every eligible provider below floor" in decision.rationale
    # anthropic 10/15=0.667, google 10/18≈0.556, moonshot 10/16=0.625, openai 10/17≈0.588
    assert decision.provider == "anthropic"


def test_provider_success_floor_cold_start_treats_unseen_as_healthy() -> None:
    decision = _strategy(SuccessStats()).choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "meets floor" in decision.rationale


def test_provider_success_floor_respects_circuit_health() -> None:
    stats = SuccessStats()
    for provider in ("anthropic", "google", "moonshot", "openai"):
        for _ in range(20):
            stats.observe(provider, success=True)
    decision = _strategy(stats, unavailable={"anthropic"}).choose(_request(), _signals())

    assert decision.chosen_model != ANTHROPIC_SAFETY_MODEL
    assert decision.provider != "anthropic"


def test_provider_success_floor_registered_by_strategy_factory() -> None:
    settings = RouterSettings(provider_success_floor=0.9)
    stats = SuccessStats()
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
        success_stats=stats,
        provider_success_floor=settings.provider_success_floor,
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_SUCCESS_FLOOR]
    assert isinstance(strategy, ProviderSuccessFloorStrategy)
    assert strategy._success_floor == 0.9  # noqa: SLF001
    assert strategy._success_stats is stats  # noqa: SLF001
    assert RouterSettings().provider_success_floor == 0.85
