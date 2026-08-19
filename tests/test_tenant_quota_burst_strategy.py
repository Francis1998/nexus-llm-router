"""Tests for tenant-quota-burst routing."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_BALANCED_MODEL
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
    TenantQuotaBurstExceededError,
    TenantQuotaBurstStats,
    TenantQuotaBurstStrategy,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(tenant: str = "tenant-a") -> RouterRequest:
    return RouterRequest(
        request_id=f"req-{tenant}",
        messages=[ChatMessage(content="Route this tenant request")],
        metadata={"tenant_id": tenant},
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=100,
    )


def _strategy(
    stats: TenantQuotaBurstStats | None = None,
    *,
    soft: int = 2,
    hard: int = 3,
) -> TenantQuotaBurstStrategy:
    return TenantQuotaBurstStrategy(
        default_model_catalog(),
        stats or TenantQuotaBurstStats(),
        soft_quota=soft,
        hard_quota=hard,
    )


def test_tenant_quota_burst_enum_parses() -> None:
    assert RoutingStrategyName("tenant-quota-burst") is RoutingStrategyName.TENANT_QUOTA_BURST


def test_tenant_quota_burst_keeps_quality_first_below_soft_quota() -> None:
    stats = TenantQuotaBurstStats()

    decision = _strategy(stats).choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.TENANT_QUOTA_BURST
    assert stats.usage("tenant-a") == 1
    assert "steady usage 1/2" in decision.rationale


def test_tenant_quota_burst_admits_soft_overage_on_cheapest_fallback() -> None:
    stats = TenantQuotaBurstStats()
    stats.record("tenant-a")
    stats.record("tenant-a")

    decision = _strategy(stats).choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert stats.usage("tenant-a") == 3
    assert "admitted burst request 3/3" in decision.rationale
    assert "shed to cheapest fallback" in decision.rationale


def test_tenant_quota_burst_rejects_without_consuming_past_hard_ceiling() -> None:
    stats = TenantQuotaBurstStats()
    for _ in range(3):
        stats.record("tenant-a")

    with pytest.raises(TenantQuotaBurstExceededError, match="hard ceiling.*shed before dispatch"):
        _strategy(stats).choose(_request(), _signals())

    assert stats.usage("tenant-a") == 3


def test_tenant_quota_burst_isolates_tenants_and_expires_old_requests() -> None:
    stats = TenantQuotaBurstStats(window_seconds=60.0)
    stats.record("tenant-a", now=10.0)
    stats.record("tenant-a", now=20.0)

    assert stats.usage("tenant-b", now=20.0) == 0
    assert stats.usage("tenant-a", now=70.0) == 2
    assert stats.usage("tenant-a", now=81.0) == 0


def test_tenant_quota_burst_rejects_invalid_limits_and_window() -> None:
    with pytest.raises(ValueError, match="soft_quota must be >= 1"):
        _strategy(soft=0)
    with pytest.raises(ValueError, match="greater than soft_quota"):
        _strategy(soft=3, hard=3)
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        TenantQuotaBurstStats(window_seconds=0.0)


def test_tenant_quota_burst_registered_by_strategy_factory() -> None:
    settings = RouterSettings(
        tenant_quota_burst_soft=40,
        tenant_quota_burst_hard=50,
        tenant_quota_burst_window_seconds=90.0,
    )
    stats = TenantQuotaBurstStats(settings.tenant_quota_burst_window_seconds)
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
        tenant_quota_burst_stats=stats,
        tenant_quota_burst_soft=settings.tenant_quota_burst_soft,
        tenant_quota_burst_hard=settings.tenant_quota_burst_hard,
        tenant_quota_burst_window_seconds=settings.tenant_quota_burst_window_seconds,
    )

    strategy = strategies[RoutingStrategyName.TENANT_QUOTA_BURST]
    assert isinstance(strategy, TenantQuotaBurstStrategy)
    assert strategy._quota_stats is stats  # noqa: SLF001
    assert strategy._soft_quota == 40  # noqa: SLF001
    assert RouterSettings().tenant_quota_burst_hard == 75
