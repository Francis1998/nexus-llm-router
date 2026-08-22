"""Tests for deadline-aware-pick routing."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, GEMINI_PRO_MODEL
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    DeadlineAwarePickStrategy,
    InflightStats,
    LatencyStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _request(**metadata: float | str) -> RouterRequest:
    return RouterRequest(
        request_id="req-deadline",
        messages=[ChatMessage(content="Route with deadline awareness.")],
        metadata=dict(metadata),
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _latencies() -> LatencyStats:
    stats = LatencyStats()
    stats.observe("anthropic", 500.0)
    stats.observe("google", 50.0)
    stats.observe("moonshot", 100.0)
    stats.observe("openai", 200.0)
    return stats


def _strategy(
    threshold_ms: float = 500.0,
    *,
    unavailable: set[str] | None = None,
) -> DeadlineAwarePickStrategy:
    return DeadlineAwarePickStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        _latencies(),
        deadline_threshold_ms=threshold_ms,
    )


def test_deadline_aware_pick_enum_parses() -> None:
    assert RoutingStrategyName("deadline-aware-pick") is RoutingStrategyName.DEADLINE_AWARE_PICK


def test_deadline_aware_pick_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="deadline_threshold_ms must be >= 0.0"):
        _strategy(-1.0)


def test_deadline_aware_pick_tight_deadline_selects_fastest_healthy() -> None:
    decision = _strategy(unavailable={"openai"}).choose(
        _request(remaining_ms=100),
        _signals(),
    )

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert decision.provider == "google"
    assert "fastest healthy deadline route" in decision.rationale
    assert "below threshold" in decision.rationale


def test_deadline_aware_pick_accepts_deadline_ms_alias() -> None:
    decision = _strategy(unavailable={"openai"}).choose(
        _request(deadline_ms=250),
        _signals(),
    )

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "below threshold" in decision.rationale


def test_deadline_aware_pick_relaxed_deadline_keeps_quality() -> None:
    decision = _strategy().choose(_request(remaining_ms=2000), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first route" in decision.rationale
    assert "at/above threshold" in decision.rationale


def test_deadline_aware_pick_missing_deadline_keeps_quality() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "no deadline metadata" in decision.rationale


def test_deadline_aware_pick_invalid_deadline_keeps_quality() -> None:
    decision = _strategy().choose(_request(remaining_ms="soon"), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "no deadline metadata" in decision.rationale


def test_deadline_aware_pick_threshold_boundary_keeps_quality() -> None:
    decision = _strategy(threshold_ms=500.0).choose(
        _request(remaining_ms=500),
        _signals(),
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "at/above threshold" in decision.rationale


def test_deadline_aware_pick_registered_by_strategy_factory() -> None:
    settings = RouterSettings(deadline_aware_threshold_ms=250.0)
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
        deadline_aware_threshold_ms=settings.deadline_aware_threshold_ms,
    )

    strategy = strategies[RoutingStrategyName.DEADLINE_AWARE_PICK]
    assert isinstance(strategy, DeadlineAwarePickStrategy)
    assert strategy._deadline_threshold_ms == 250.0  # noqa: SLF001
    assert RouterSettings().deadline_aware_threshold_ms == 500.0
