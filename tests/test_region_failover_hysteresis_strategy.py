"""Tests for the region-failover-hysteresis routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import MOONSHOT_BALANCED_MODEL
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
    RegionFailoverHysteresisStats,
    RegionFailoverHysteresisStrategy,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _request(
    *,
    session_id: str = "session-hyst",
    region: str | None = "eu",
) -> RouterRequest:
    return RouterRequest(
        request_id="req-region-hyst",
        session_id=session_id,
        messages=[ChatMessage(content="Hello")],
        region=region,
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=64,
    )


def _strategy(
    *,
    unavailable: set[str] | None = None,
    hysteresis_successes: int = 3,
    stats: RegionFailoverHysteresisStats | None = None,
) -> RegionFailoverHysteresisStrategy:
    return RegionFailoverHysteresisStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        stats or RegionFailoverHysteresisStats(),
        region_preferences=["eu", "us", "cn", "global"],
        hysteresis_successes=hysteresis_successes,
    )


def test_region_failover_hysteresis_enum_parses() -> None:
    assert (
        RoutingStrategyName("region-failover-hysteresis")
        is RoutingStrategyName.REGION_FAILOVER_HYSTERESIS
    )


def test_region_failover_hysteresis_prefers_requested_region() -> None:
    decision = _strategy().choose(_request(region="cn"), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "preferred region healthy" in decision.rationale


def test_region_failover_hysteresis_failovers_when_preferred_unhealthy() -> None:
    decision = _strategy(unavailable={"anthropic", "google"}).choose(
        _request(region="eu"),
        _signals(),
    )

    assert decision.provider == "openai"
    assert "failover from preferred 'eu'" in decision.rationale


def test_region_failover_hysteresis_holds_failover_until_recovery_streak() -> None:
    stats = RegionFailoverHysteresisStats()
    strategy = _strategy(unavailable={"anthropic", "google"}, stats=stats)
    strategy.choose(_request(session_id="hold-session", region="eu"), _signals())

    stats.record_success("eu")
    stats.record_success("eu")
    decision = strategy.choose(_request(session_id="hold-session", region="eu"), _signals())

    assert decision.provider == "openai"
    assert "holding failover until preferred region reaches hysteresis 2/3" in decision.rationale


def test_region_failover_hysteresis_returns_after_recovery_streak() -> None:
    stats = RegionFailoverHysteresisStats()
    health = _FakeHealth({"anthropic", "google"})
    strategy = RegionFailoverHysteresisStrategy(
        default_model_catalog(),
        health,
        stats,
        region_preferences=["eu", "us", "cn", "global"],
        hysteresis_successes=3,
    )
    strategy.choose(_request(session_id="recover-session", region="eu"), _signals())

    for _ in range(3):
        stats.record_success("eu")
    health._unavailable.clear()
    decision = strategy.choose(_request(session_id="recover-session", region="eu"), _signals())

    assert decision.provider == "google"
    assert "preferred region recovered with 3/3 consecutive successes" in decision.rationale


def test_region_failover_hysteresis_rejects_non_positive_successes() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        RegionFailoverHysteresisStrategy(
            default_model_catalog(),
            _FakeHealth(),
            RegionFailoverHysteresisStats(),
            hysteresis_successes=0,
        )


def test_region_failover_hysteresis_registered_by_strategy_factory() -> None:
    settings = RouterSettings(region_failover_hysteresis_successes=5)
    stats = RegionFailoverHysteresisStats()
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
        region_failover_hysteresis_successes=settings.region_failover_hysteresis_successes,
        region_failover_hysteresis_stats=stats,
    )

    strategy = strategies[RoutingStrategyName.REGION_FAILOVER_HYSTERESIS]
    assert isinstance(strategy, RegionFailoverHysteresisStrategy)
    assert strategy._hysteresis_successes == 5  # noqa: SLF001
    assert RouterSettings().region_failover_hysteresis_successes == 3
