"""Tests for tenant-soft-isolation routing."""

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
    SuccessStats,
    TenantSoftIsolationStrategy,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _request(metadata: dict | None = None) -> RouterRequest:
    return RouterRequest(
        request_id="req-tenant-soft-isolation",
        messages=[ChatMessage(content="Route with tenant soft isolation.")],
        metadata=metadata or {},
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
    *,
    soft_rpm: int = 60,
    unavailable: set[str] | None = None,
) -> TenantSoftIsolationStrategy:
    return TenantSoftIsolationStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        soft_isolation_rpm=soft_rpm,
    )


def _cheapest_model() -> str:
    catalog = default_model_catalog()
    cheapest = min(
        catalog.values(),
        key=lambda candidate: candidate.estimate_cost(128, 512),
    )
    return cheapest.model


def test_tenant_soft_isolation_enum_parses() -> None:
    assert RoutingStrategyName("tenant-soft-isolation") is RoutingStrategyName.TENANT_SOFT_ISOLATION


def test_tenant_soft_isolation_rejects_invalid_rpm() -> None:
    with pytest.raises(ValueError, match="soft_isolation_rpm must be >= 1"):
        _strategy(soft_rpm=0)


def test_tenant_soft_isolation_stays_quality_first_without_metadata() -> None:
    decision = _strategy(soft_rpm=60).choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.TENANT_SOFT_ISOLATION


def test_tenant_soft_isolation_stays_quality_first_at_or_below_threshold() -> None:
    decision = _strategy(soft_rpm=60).choose(_request({"tenant_rpm": 60}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_tenant_soft_isolation_shifts_to_lowest_cost_above_threshold() -> None:
    decision = _strategy(soft_rpm=60).choose(_request({"tenant_rpm": 61}), _signals())

    assert decision.chosen_model == _cheapest_model()
    assert "soft-isolated" in decision.rationale
    assert "exceeded soft rate" in decision.rationale


def test_tenant_soft_isolation_reads_tenant_request_rate_alias() -> None:
    decision = _strategy(soft_rpm=60).choose(_request({"tenant_request_rate": 90}), _signals())

    assert decision.chosen_model == _cheapest_model()


def test_tenant_soft_isolation_prefers_tenant_rpm_over_alias() -> None:
    decision = _strategy(soft_rpm=60).choose(
        _request({"tenant_rpm": 5, "tenant_request_rate": 90}), _signals()
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_tenant_soft_isolation_ignores_malformed_rate() -> None:
    decision = _strategy(soft_rpm=60).choose(_request({"tenant_rpm": "not-a-number"}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_tenant_soft_isolation_falls_back_to_user_id_without_tenant_metadata() -> None:
    strategy = _strategy(soft_rpm=60)
    request = RouterRequest(
        request_id="req-user-fallback",
        messages=[ChatMessage(content="hi")],
        user_id="user-123",
    )
    decision = strategy.choose(request, _signals())
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "user-123" in decision.rationale


def test_tenant_soft_isolation_respects_circuit_health() -> None:
    decision = _strategy(unavailable={"anthropic"}).choose(_request(), _signals())
    assert decision.provider != "anthropic"


def test_tenant_soft_isolation_registered_by_strategy_factory() -> None:
    settings = RouterSettings(tenant_soft_isolation_rpm=45)
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
        success_stats=SuccessStats(),
        tenant_soft_isolation_rpm=settings.tenant_soft_isolation_rpm,
    )

    strategy = strategies[RoutingStrategyName.TENANT_SOFT_ISOLATION]
    assert isinstance(strategy, TenantSoftIsolationStrategy)
    assert strategy._soft_isolation_rpm == 45  # noqa: SLF001
    assert RouterSettings().tenant_soft_isolation_rpm == 60
