"""Tests for sticky-region-warmup routing."""

import pytest

from router.config import RouterSettings, default_model_catalog
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
    StickyRegionWarmupStats,
    StickyRegionWarmupStrategy,
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
    session_id: str = "session-warmup",
    region: str | None = "us",
    warmup_region: str | None = None,
) -> RouterRequest:
    metadata = {} if warmup_region is None else {"warmup_region": warmup_region}
    return RouterRequest(
        request_id="req-sticky-warmup",
        session_id=session_id,
        messages=[ChatMessage(content="Keep this session region stable")],
        region=region,
        metadata=metadata,
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
    stats: StickyRegionWarmupStats | None = None,
    *,
    unavailable: set[str] | None = None,
    warmup_requests: int = 2,
) -> StickyRegionWarmupStrategy:
    return StickyRegionWarmupStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        stats or StickyRegionWarmupStats(),
        region_preferences=["eu", "us", "cn", "global"],
        warmup_request_count=warmup_requests,
    )


def test_sticky_region_warmup_enum_parses() -> None:
    assert RoutingStrategyName("sticky-region-warmup") is RoutingStrategyName.STICKY_REGION_WARMUP


def test_sticky_region_warmup_uses_warmup_region_for_new_session() -> None:
    decision = _strategy().choose(_request(), _signals())
    candidate = default_model_catalog()[decision.chosen_model]

    assert "eu" in candidate.supported_regions
    assert decision.routing_strategy is RoutingStrategyName.STICKY_REGION_WARMUP
    assert "warmup request 1/2" in decision.rationale


def test_sticky_region_warmup_pins_requested_region_after_warmup() -> None:
    stats = StickyRegionWarmupStats()
    strategy = _strategy(stats)
    strategy.choose(_request(), _signals())
    strategy.choose(_request(), _signals())

    decision = strategy.choose(_request(region="us"), _signals())
    candidate = default_model_catalog()[decision.chosen_model]

    assert "us" in candidate.supported_regions
    assert stats.pinned_region("session-warmup") == "us"
    assert "completed 2 warmup requests" in decision.rationale


def test_sticky_region_warmup_keeps_pin_when_request_region_changes() -> None:
    stats = StickyRegionWarmupStats()
    strategy = _strategy(stats, warmup_requests=1)
    strategy.choose(_request(region="us"), _signals())
    pinned = strategy.choose(_request(region="us"), _signals())

    changed = strategy.choose(_request(region="cn"), _signals())

    assert changed.chosen_model == pinned.chosen_model
    assert stats.pinned_region("session-warmup") == "us"
    assert "region 'us'" in changed.rationale


def test_sticky_region_warmup_uses_healthy_fallback_if_warmup_region_is_down() -> None:
    decision = _strategy(unavailable={"anthropic", "google"}).choose(
        _request(warmup_region="eu"),
        _signals(),
    )

    assert decision.provider in {"openai", "moonshot"}
    assert "region 'eu' unavailable" in decision.rationale


def test_sticky_region_warmup_rejects_non_positive_request_count() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        _strategy(warmup_requests=0)


def test_sticky_region_warmup_registered_by_strategy_factory() -> None:
    settings = RouterSettings(sticky_region_warmup_requests=5)
    stats = StickyRegionWarmupStats()
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
        sticky_region_warmup_requests=settings.sticky_region_warmup_requests,
        sticky_region_warmup_stats=stats,
    )

    strategy = strategies[RoutingStrategyName.STICKY_REGION_WARMUP]
    assert isinstance(strategy, StickyRegionWarmupStrategy)
    assert strategy._warmup_stats is stats  # noqa: SLF001
    assert strategy._warmup_request_count == 5  # noqa: SLF001
    assert RouterSettings().sticky_region_warmup_requests == 3
