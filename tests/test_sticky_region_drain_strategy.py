"""Tests for sticky-region-drain routing."""

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
    StickyRegionDrainStats,
    StickyRegionDrainStrategy,
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
    session_id: str = "session-drain",
    region: str = "us",
    draining_regions: list[str] | None = None,
) -> RouterRequest:
    metadata = {} if draining_regions is None else {"draining_regions": draining_regions}
    return RouterRequest(
        request_id=f"req-{session_id}",
        session_id=session_id,
        messages=[ChatMessage(content="Keep this regional session available.")],
        region=region,
        metadata=metadata,
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _strategy(
    stats: StickyRegionDrainStats | None = None,
    *,
    unavailable: set[str] | None = None,
    draining_regions: list[str] | None = None,
) -> StickyRegionDrainStrategy:
    return StickyRegionDrainStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        stats or StickyRegionDrainStats(),
        region_preferences=["us", "eu", "cn", "global"],
        draining_regions=draining_regions,
    )


def test_sticky_region_drain_enum_parses() -> None:
    assert RoutingStrategyName("sticky-region-drain") is RoutingStrategyName.STICKY_REGION_DRAIN


def test_sticky_region_drain_pins_requested_healthy_region() -> None:
    stats = StickyRegionDrainStats()

    decision = _strategy(stats).choose(_request(), _signals())

    assert stats.pinned_region("session-drain") == "us"
    assert "us" in default_model_catalog()[decision.chosen_model].supported_regions
    assert decision.routing_strategy is RoutingStrategyName.STICKY_REGION_DRAIN
    assert "healthy non-draining region 'us'" in decision.rationale


def test_sticky_region_drain_keeps_pin_when_request_region_changes() -> None:
    stats = StickyRegionDrainStats()
    strategy = _strategy(stats)
    first = strategy.choose(_request(region="us"), _signals())

    changed = strategy.choose(_request(region="cn"), _signals())

    assert changed.chosen_model == first.chosen_model
    assert stats.pinned_region("session-drain") == "us"
    assert stats.migration_count("session-drain") == 0


def test_sticky_region_drain_evacuates_configured_region() -> None:
    stats = StickyRegionDrainStats()

    decision = _strategy(stats, draining_regions=["us"]).choose(
        _request(region="us"),
        _signals(),
    )

    assert stats.pinned_region("session-drain") == "eu"
    assert "eu" in default_model_catalog()[decision.chosen_model].supported_regions
    assert "away from marked region 'us' to healthy alternate 'eu'" in decision.rationale


def test_sticky_region_drain_metadata_migrates_existing_pin_once() -> None:
    stats = StickyRegionDrainStats()
    strategy = _strategy(stats)
    strategy.choose(_request(region="us"), _signals())

    drained = strategy.choose(
        _request(region="us", draining_regions=["us"]),
        _signals(),
    )
    stable = strategy.choose(_request(region="us"), _signals())

    assert stats.pinned_region("session-drain") == "eu"
    assert stats.migration_count("session-drain") == 1
    assert stable.chosen_model == drained.chosen_model


def test_sticky_region_drain_skips_unhealthy_first_alternate() -> None:
    stats = StickyRegionDrainStats()

    decision = _strategy(
        stats,
        unavailable={"anthropic", "google"},
        draining_regions=["us"],
    ).choose(_request(), _signals())

    assert stats.pinned_region("session-drain") == "cn"
    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert decision.provider == "moonshot"


def test_sticky_region_drain_retains_region_if_every_region_is_draining() -> None:
    stats = StickyRegionDrainStats()

    decision = _strategy(
        stats,
        draining_regions=["us", "eu", "cn", "global"],
    ).choose(_request(), _signals())

    assert stats.pinned_region("session-drain") == "us"
    assert "us" in default_model_catalog()[decision.chosen_model].supported_regions
    assert "no healthy non-draining alternate exists" in decision.rationale


def test_sticky_region_drain_uses_emergency_fallback_without_healthy_alternate() -> None:
    stats = StickyRegionDrainStats()

    decision = _strategy(
        stats,
        unavailable={"anthropic", "openai", "google", "moonshot"},
        draining_regions=["us"],
    ).choose(_request(), _signals())

    assert stats.pinned_region("session-drain") == "us"
    assert decision.chosen_model in default_model_catalog()
    assert "no healthy non-draining alternate exists" in decision.rationale


def test_sticky_region_drain_registered_by_strategy_factory() -> None:
    settings = RouterSettings(sticky_region_drain_regions=["us"])
    stats = StickyRegionDrainStats()
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
        sticky_region_drain_stats=stats,
        sticky_region_drain_regions=settings.sticky_region_drain_regions,
    )

    strategy = strategies[RoutingStrategyName.STICKY_REGION_DRAIN]
    assert isinstance(strategy, StickyRegionDrainStrategy)
    assert strategy._drain_stats is stats  # noqa: SLF001
    assert strategy._draining_regions == {"us"}  # noqa: SLF001
    assert RouterSettings().sticky_region_drain_regions == []
