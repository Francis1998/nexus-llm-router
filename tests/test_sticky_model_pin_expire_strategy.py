"""Tests for sticky-model-pin-expire routing."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    OPENAI_BALANCED_MODEL,
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
    InflightStats,
    LatencyStats,
    StickyModelPinExpireStats,
    StickyModelPinExpireStrategy,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self.unavailable = unavailable if unavailable is not None else set()

    def is_available(self, provider: str) -> bool:
        return provider not in self.unavailable


def _request(session_id: str = "session-pin", request_id: str = "req-pin") -> RouterRequest:
    return RouterRequest(
        request_id=request_id,
        session_id=session_id,
        messages=[ChatMessage(content="Keep model affinity until expiry.")],
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
    stats: StickyModelPinExpireStats | None = None,
    *,
    health: _FakeHealth | None = None,
    ttl_seconds: float = 300.0,
) -> StickyModelPinExpireStrategy:
    return StickyModelPinExpireStrategy(
        default_model_catalog(),
        health or _FakeHealth(),
        stats or StickyModelPinExpireStats(),
        ttl_seconds=ttl_seconds,
    )


def test_sticky_model_pin_expire_enum_parses() -> None:
    assert (
        RoutingStrategyName("sticky-model-pin-expire")
        is RoutingStrategyName.STICKY_MODEL_PIN_EXPIRE
    )


def test_sticky_model_pin_expire_stats_rejects_invalid_ttl() -> None:
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        StickyModelPinExpireStats().pin("session", "model", 0.0)


def test_sticky_model_pin_expire_stats_removes_pin_at_deadline() -> None:
    stats = StickyModelPinExpireStats()
    stats.pin("session", OPENAI_BALANCED_MODEL, 5.0, now=10.0)

    assert stats.pinned_model("session", now=14.0) == OPENAI_BALANCED_MODEL
    assert stats.remaining_seconds("session", now=14.0) == 1.0
    assert stats.pinned_model("session", now=15.0) is None
    assert stats.expiration_count("session") == 1


def test_sticky_model_pin_expire_creates_quality_first_pin() -> None:
    stats = StickyModelPinExpireStats()
    decision = _strategy(stats, ttl_seconds=30.0).choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert stats.pinned_model("session-pin") == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.STICKY_MODEL_PIN_EXPIRE
    assert "created a 30.00s pin" in decision.rationale


def test_sticky_model_pin_expire_keeps_unexpired_model() -> None:
    stats = StickyModelPinExpireStats()
    stats.pin("session-pin", OPENAI_BALANCED_MODEL, 30.0)

    decision = _strategy(stats).choose(
        _request(request_id="different-request"),
        _signals(),
    )

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "kept session 'session-pin' on unexpired pin" in decision.rationale


def test_sticky_model_pin_expire_rechooses_after_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr("router.strategies.time.monotonic", lambda: clock[0])
    stats = StickyModelPinExpireStats()
    health = _FakeHealth()
    strategy = _strategy(stats, health=health, ttl_seconds=5.0)
    first = strategy.choose(_request(), _signals())
    health.unavailable.add(first.provider)
    clock[0] = 5.0

    second = strategy.choose(_request(request_id="req-after-expiry"), _signals())

    assert first.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert second.chosen_model == OPENAI_FRONTIER_MODEL
    assert second.chosen_model != first.chosen_model
    assert "model pin TTL expired" in second.rationale
    assert stats.expiration_count("session-pin") == 1


def test_sticky_model_pin_expire_reselects_unhealthy_pin_before_ttl() -> None:
    stats = StickyModelPinExpireStats()
    stats.pin("session-pin", ANTHROPIC_SAFETY_MODEL, 300.0)
    decision = _strategy(
        stats,
        health=_FakeHealth({"anthropic"}),
    ).choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert "pinned provider anthropic became unavailable before TTL" in decision.rationale
    assert stats.expiration_count("session-pin") == 0


def test_sticky_model_pin_expire_tracks_sessions_independently() -> None:
    stats = StickyModelPinExpireStats()
    stats.pin("session-a", OPENAI_BALANCED_MODEL, 300.0)
    stats.pin("session-b", ANTHROPIC_SAFETY_MODEL, 300.0)
    strategy = _strategy(stats)

    first = strategy.choose(_request("session-a"), _signals())
    second = strategy.choose(_request("session-b"), _signals())

    assert first.chosen_model == OPENAI_BALANCED_MODEL
    assert second.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_sticky_model_pin_expire_rejects_invalid_strategy_ttl() -> None:
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        _strategy(ttl_seconds=0.0)


def test_sticky_model_pin_expire_registered_by_strategy_factory() -> None:
    settings = RouterSettings(sticky_model_pin_ttl_seconds=17.0)
    stats = StickyModelPinExpireStats()
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
        sticky_model_pin_expire_stats=stats,
        sticky_model_pin_ttl_seconds=settings.sticky_model_pin_ttl_seconds,
    )

    strategy = strategies[RoutingStrategyName.STICKY_MODEL_PIN_EXPIRE]
    assert isinstance(strategy, StickyModelPinExpireStrategy)
    assert strategy._pin_stats is stats  # noqa: SLF001
    assert strategy._ttl_seconds == 17.0  # noqa: SLF001
    assert RouterSettings().sticky_model_pin_ttl_seconds == 300.0
