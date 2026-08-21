"""Tests for tenant-priority-lanes routing."""

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
    InflightStats,
    LatencyStats,
    TenantPriorityLane,
    TenantPriorityLanesStrategy,
    TenantPriorityLaneStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _request(tenant: str = "tenant-a", lane: str | None = None) -> RouterRequest:
    metadata = {"tenant_id": tenant}
    if lane is not None:
        metadata["priority_lane"] = lane
    return RouterRequest(
        request_id=f"req-{tenant}",
        messages=[ChatMessage(content="Route this tenant by priority.")],
        metadata=metadata,
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
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
    stats: TenantPriorityLaneStats | None = None,
    *,
    unavailable: set[str] | None = None,
    high_tenants: list[str] | None = None,
    low_tenants: list[str] | None = None,
    high_quota: int = 100,
    normal_quota: int = 60,
    low_quota: int = 30,
) -> TenantPriorityLanesStrategy:
    return TenantPriorityLanesStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        _latencies(),
        stats or TenantPriorityLaneStats(),
        high_tenants=high_tenants,
        low_tenants=low_tenants,
        high_quota=high_quota,
        normal_quota=normal_quota,
        low_quota=low_quota,
    )


def test_tenant_priority_lanes_enum_parses() -> None:
    assert RoutingStrategyName("tenant-priority-lanes") is RoutingStrategyName.TENANT_PRIORITY_LANES


def test_tenant_priority_lane_stats_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="lookback must be >= 1"):
        TenantPriorityLaneStats(0)
    with pytest.raises(ValueError, match="quota must be >= 1"):
        TenantPriorityLaneStats().at_quota(TenantPriorityLane.HIGH, 0)


def test_tenant_priority_lane_stats_evicts_stale_decisions() -> None:
    stats = TenantPriorityLaneStats(lookback=2)
    stats.observe(TenantPriorityLane.HIGH)
    stats.observe(TenantPriorityLane.NORMAL)
    stats.observe(TenantPriorityLane.NORMAL)

    assert stats.lane_count(TenantPriorityLane.HIGH) == 0
    assert stats.lane_count(TenantPriorityLane.NORMAL) == 2


def test_tenant_priority_lanes_high_tenant_gets_fastest_healthy_route() -> None:
    decision = _strategy(
        unavailable={"openai"},
        high_tenants=["critical"],
    ).choose(_request("critical"), _signals())

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert decision.provider == "google"
    assert "high lane" in decision.rationale
    assert "fastest observed healthy priority route" in decision.rationale


def test_tenant_priority_lanes_normal_tenant_keeps_quality_under_pressure() -> None:
    decision = _strategy(unavailable={"openai"}).choose(_request("standard"), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "normal lane" in decision.rationale
    assert "quality-first route" in decision.rationale


def test_tenant_priority_lanes_request_metadata_overrides_mapping() -> None:
    decision = _strategy(
        unavailable={"openai"},
        high_tenants=["critical"],
    ).choose(_request("critical", lane="low"), _signals())

    candidates = [
        candidate
        for candidate in default_model_catalog().values()
        if candidate.provider != "openai" and DomainTag.GENERAL in candidate.supports_domains
    ]
    cheapest = min(
        candidates,
        key=lambda candidate: (
            candidate.estimate_cost(_signals().prompt_tokens_estimate, 512),
            -candidate.quality_score,
            candidate.model,
        ),
    )
    assert decision.chosen_model == cheapest.model
    assert "low lane" in decision.rationale


def test_tenant_priority_lanes_low_quota_uses_relief_route() -> None:
    stats = TenantPriorityLaneStats()
    stats.observe(TenantPriorityLane.LOW)
    decision = _strategy(
        stats,
        low_tenants=["batch"],
        low_quota=1,
    ).choose(_request("batch"), _signals())

    eligible = [
        candidate
        for candidate in default_model_catalog().values()
        if DomainTag.GENERAL in candidate.supports_domains
    ]
    cheapest = min(
        eligible,
        key=lambda candidate: (
            candidate.estimate_cost(_signals().prompt_tokens_estimate, 512),
            -candidate.quality_score,
            candidate.model,
        ),
    )
    assert decision.chosen_model == cheapest.model
    assert "low lane reached 1 recent decisions" in decision.rationale
    assert "cost-efficient relief route" in decision.rationale


def test_tenant_priority_lanes_unconstrained_low_tenant_keeps_quality() -> None:
    decision = _strategy(low_tenants=["batch"]).choose(_request("batch"), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "capacity available" in decision.rationale
    assert "quality-first route" in decision.rationale


def test_tenant_priority_lanes_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="high_quota must be >= 1"):
        _strategy(high_quota=0)
    with pytest.raises(ValueError, match="both high and low lanes"):
        _strategy(high_tenants=["duplicate"], low_tenants=["duplicate"])


def test_tenant_priority_lanes_registered_by_strategy_factory() -> None:
    settings = RouterSettings(
        tenant_priority_high_tenants=["critical"],
        tenant_priority_low_tenants=["batch"],
        tenant_priority_lane_lookback=17,
        tenant_priority_high_quota=12,
        tenant_priority_normal_quota=8,
        tenant_priority_low_quota=4,
    )
    stats = TenantPriorityLaneStats(settings.tenant_priority_lane_lookback)
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
        tenant_priority_lane_stats=stats,
        tenant_priority_high_tenants=settings.tenant_priority_high_tenants,
        tenant_priority_low_tenants=settings.tenant_priority_low_tenants,
        tenant_priority_lane_lookback=settings.tenant_priority_lane_lookback,
        tenant_priority_high_quota=settings.tenant_priority_high_quota,
        tenant_priority_normal_quota=settings.tenant_priority_normal_quota,
        tenant_priority_low_quota=settings.tenant_priority_low_quota,
    )

    strategy = strategies[RoutingStrategyName.TENANT_PRIORITY_LANES]
    assert isinstance(strategy, TenantPriorityLanesStrategy)
    assert strategy._lane_stats is stats  # noqa: SLF001
    assert strategy._high_tenants == {"critical"}  # noqa: SLF001
    assert strategy._low_tenants == {"batch"}  # noqa: SLF001
    assert RouterSettings().tenant_priority_lane_lookback == 100
