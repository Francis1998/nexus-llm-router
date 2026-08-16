"""Tests for the tenant-budget-cascade routing strategy."""

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
    TenantBudgetCascadeStats,
    TenantBudgetCascadeStrategy,
    TenantBudgetExceededError,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(tenant: str = "tenant-a") -> RouterRequest:
    return RouterRequest(
        request_id="req-tenant-budget",
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
    stats: TenantBudgetCascadeStats | None = None,
    *,
    soft: float = 0.01,
    hard: float = 0.02,
) -> TenantBudgetCascadeStrategy:
    return TenantBudgetCascadeStrategy(
        default_model_catalog(),
        stats or TenantBudgetCascadeStats(),
        soft_budget=soft,
        hard_budget=hard,
    )


def test_tenant_budget_cascade_enum_parses() -> None:
    assert RoutingStrategyName("tenant-budget-cascade") is RoutingStrategyName.TENANT_BUDGET_CASCADE


def test_tenant_budget_cascade_is_quality_first_with_soft_headroom() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.TENANT_BUDGET_CASCADE
    assert "under soft" in decision.rationale


def test_tenant_budget_cascade_sheds_to_cheapest_near_hard_ceiling() -> None:
    stats = TenantBudgetCascadeStats()
    stats.record_spend("tenant-a", 0.0097)

    decision = _strategy(stats).choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert decision.provider == "openai"
    assert "near hard ceiling" in decision.rationale


def test_tenant_budget_cascade_fails_closed_before_crossing_hard_ceiling() -> None:
    stats = TenantBudgetCascadeStats()
    stats.record_spend("tenant-a", 0.0198)

    with pytest.raises(TenantBudgetExceededError, match="hard ceiling.*fail closed"):
        _strategy(stats).choose(_request(), _signals())


def test_tenant_budget_cascade_keeps_tenant_windows_isolated() -> None:
    stats = TenantBudgetCascadeStats()
    stats.record_spend("tenant-a", 0.0198)

    decision = _strategy(stats).choose(_request("tenant-b"), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert stats.spend("tenant-b") == 0.0


def test_tenant_budget_cascade_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="greater than soft_budget"):
        _strategy(soft=1.0, hard=1.0)


def test_tenant_budget_cascade_registered_by_strategy_factory() -> None:
    settings = RouterSettings(
        tenant_budget_cascade_soft=7.0,
        tenant_budget_cascade_hard=9.0,
    )
    stats = TenantBudgetCascadeStats()
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
        tenant_budget_cascade_soft=settings.tenant_budget_cascade_soft,
        tenant_budget_cascade_hard=settings.tenant_budget_cascade_hard,
        tenant_budget_cascade_stats=stats,
    )

    strategy = strategies[RoutingStrategyName.TENANT_BUDGET_CASCADE]
    assert isinstance(strategy, TenantBudgetCascadeStrategy)
    assert strategy._tenant_budget_stats is stats  # noqa: SLF001
    assert RouterSettings().tenant_budget_cascade_soft == 10.0
