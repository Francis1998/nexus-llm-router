"""Tests for tenant-fair-queue routing."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_BALANCED_MODEL,
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
    TenantFairQueueStats,
    TenantFairQueueStrategy,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _request(tenant: str = "tenant-a") -> RouterRequest:
    return RouterRequest(
        request_id=f"req-{tenant}",
        messages=[ChatMessage(content="Route this tenant fairly.")],
        metadata={"tenant_id": tenant},
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
    stats: TenantFairQueueStats | None = None,
    *,
    unavailable: set[str] | None = None,
) -> TenantFairQueueStrategy:
    return TenantFairQueueStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        stats or TenantFairQueueStats(),
    )


def test_tenant_fair_queue_enum_parses() -> None:
    assert RoutingStrategyName("tenant-fair-queue") is RoutingStrategyName.TENANT_FAIR_QUEUE


def test_tenant_fair_queue_stats_rejects_invalid_lookback() -> None:
    with pytest.raises(ValueError, match="lookback must be >= 1"):
        TenantFairQueueStats(0)


def test_tenant_fair_queue_stats_evicts_stale_requests() -> None:
    stats = TenantFairQueueStats(lookback=2)
    stats.observe("tenant-a")
    stats.observe("tenant-b")
    stats.observe("tenant-b")

    assert stats.request_count("tenant-a") == 0
    assert stats.request_count("tenant-b") == 2


def test_tenant_fair_queue_cold_start_uses_quality_lane() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.TENANT_FAIR_QUEUE
    assert "deficit priority" in decision.rationale


def test_tenant_fair_queue_moves_overrepresented_tenant_to_relief_lane() -> None:
    stats = TenantFairQueueStats()
    for _ in range(4):
        stats.observe("noisy")
    stats.observe("quiet")

    decision = _strategy(stats).choose(_request("noisy"), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "4 recent requests exceeds fair share 2.50" in decision.rationale
    assert "relief lane" in decision.rationale


def test_tenant_fair_queue_prioritizes_underrepresented_tenant() -> None:
    stats = TenantFairQueueStats()
    for _ in range(4):
        stats.observe("noisy")
    stats.observe("quiet")

    decision = _strategy(stats).choose(_request("quiet"), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "deficit 1.50" in decision.rationale


def test_tenant_fair_queue_relief_lane_respects_provider_health() -> None:
    stats = TenantFairQueueStats()
    for _ in range(4):
        stats.observe("noisy")
    stats.observe("quiet")

    decision = _strategy(stats, unavailable={"openai"}).choose(
        _request("noisy"),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert decision.provider == "moonshot"


def test_tenant_fair_queue_records_selected_tenant() -> None:
    stats = TenantFairQueueStats()

    _strategy(stats).choose(_request("acme"), _signals())

    assert stats.request_count("acme") == 1
    assert stats.total_requests == 1


def test_tenant_fair_queue_registered_by_strategy_factory() -> None:
    settings = RouterSettings(tenant_fair_queue_lookback=17)
    stats = TenantFairQueueStats(settings.tenant_fair_queue_lookback)
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
        tenant_fair_queue_stats=stats,
        tenant_fair_queue_lookback=settings.tenant_fair_queue_lookback,
    )

    strategy = strategies[RoutingStrategyName.TENANT_FAIR_QUEUE]
    assert isinstance(strategy, TenantFairQueueStrategy)
    assert strategy._fair_queue_stats is stats  # noqa: SLF001
    assert RouterSettings().tenant_fair_queue_lookback == 100
