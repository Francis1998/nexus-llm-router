"""Tests for the tenant-concurrency-lease routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_FRONTIER_MODEL
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
    TenantConcurrencyLeaseStrategy,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(
    *,
    metadata: dict[str, str] | None = None,
    session_id: str = "sess-lease",
) -> RouterRequest:
    """Build a router request for tenant-lease tests."""
    return RouterRequest(
        request_id="req-tenant-lease",
        messages=[ChatMessage(content="lease check")],
        metadata=metadata or {},
        session_id=session_id,
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=48,
    )


def test_tenant_concurrency_lease_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("tenant-concurrency-lease")
        is RoutingStrategyName.TENANT_CONCURRENCY_LEASE
    )


def test_tenant_concurrency_lease_cold_start_picks_top_quality() -> None:
    """With no tenant load, the highest-quality eligible model wins."""
    strategy = TenantConcurrencyLeaseStrategy(
        default_model_catalog(),
        InflightStats(),
        tenant_concurrency_lease=8,
    )

    decision = strategy.choose(_request(metadata={"tenant_id": "acme"}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "under lease 8 for tenant 'acme'" in decision.rationale
    assert "load 0/8" in decision.rationale


def test_tenant_concurrency_lease_skips_saturated_provider_for_tenant() -> None:
    """A provider at the tenant lease should not win primary selection."""
    stats = InflightStats()
    stats.begin_for_tenant("acme", "anthropic")
    strategy = TenantConcurrencyLeaseStrategy(
        default_model_catalog(),
        stats,
        tenant_concurrency_lease=1,
    )

    decision = strategy.choose(_request(metadata={"tenant_id": "acme"}), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "under lease 1 for tenant 'acme'" in decision.rationale


def test_tenant_concurrency_lease_isolates_tenants() -> None:
    """One tenant's lease load must not block another tenant."""
    stats = InflightStats()
    stats.begin_for_tenant("acme", "anthropic")
    strategy = TenantConcurrencyLeaseStrategy(
        default_model_catalog(),
        stats,
        tenant_concurrency_lease=1,
    )

    acme = strategy.choose(_request(metadata={"tenant_id": "acme"}), _signals())
    globex = strategy.choose(_request(metadata={"tenant_id": "globex"}), _signals())

    assert acme.chosen_model == OPENAI_FRONTIER_MODEL
    assert globex.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_tenant_concurrency_lease_all_leased_falls_back_to_least_loaded() -> None:
    """If every provider is at lease for the tenant, pick the least-loaded one."""
    stats = InflightStats()
    for provider in ("anthropic", "openai", "google", "moonshot"):
        stats.begin_for_tenant("acme", provider)
    stats.begin_for_tenant("acme", "anthropic")
    strategy = TenantConcurrencyLeaseStrategy(
        default_model_catalog(),
        stats,
        tenant_concurrency_lease=1,
    )

    decision = strategy.choose(_request(metadata={"tenant_id": "acme"}), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert "every eligible provider at or above lease" in decision.rationale
    assert "openai load 1/1" in decision.rationale


def test_tenant_concurrency_lease_rejects_non_positive_lease() -> None:
    """A non-positive lease fails fast at construction."""
    with pytest.raises(ValueError, match=">= 1"):
        TenantConcurrencyLeaseStrategy(
            default_model_catalog(),
            InflightStats(),
            tenant_concurrency_lease=0,
        )


def test_tenant_concurrency_lease_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose tenant-concurrency-lease."""
    settings = RouterSettings(tenant_concurrency_lease=4)
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
        settings.prompt_prefix_cache_min_chars,
        settings.epsilon,
        settings.availability_slo,
        SuccessStats(),
        settings.failover_priority,
        settings.health_blend_success_weight,
        settings.health_blend_latency_weight,
        settings.health_blend_quality_weight,
        settings.health_blend_cost_weight,
        settings.concurrency_cap,
        provider_family_cost_ceiling_usd=settings.provider_family_cost_ceiling_usd,
        cache_hit_sticky_min_chars=settings.cache_hit_sticky_min_chars,
        tenant_concurrency_lease=settings.tenant_concurrency_lease,
    )

    strategy = strategies[RoutingStrategyName.TENANT_CONCURRENCY_LEASE]
    assert isinstance(strategy, TenantConcurrencyLeaseStrategy)
    assert strategy.strategy_name is RoutingStrategyName.TENANT_CONCURRENCY_LEASE
