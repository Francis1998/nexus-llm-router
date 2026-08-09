"""Tests for the sticky-canary-cost routing strategy."""

from hashlib import sha256

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
    StickyCanaryCostStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    """Deterministic provider-health view for unit tests."""

    def __init__(self, unavailable: set[str]) -> None:
        """Store providers considered unavailable."""
        self._unavailable = unavailable

    def is_available(self, provider: str) -> bool:
        """Return whether a provider is routable."""
        return provider not in self._unavailable


def _signals() -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def _request(
    *,
    request_id: str = "req-sticky-canary",
    metadata: dict[str, str] | None = None,
    user_id: str = "anonymous",
    session_id: str = "default",
) -> RouterRequest:
    """Build a router request for sticky-canary-cost tests."""
    return RouterRequest(
        request_id=request_id,
        messages=[ChatMessage(content="hello")],
        metadata=metadata or {},
        user_id=user_id,
        session_id=session_id,
    )


def _primary_model(tenant_id: str) -> str:
    """Reproduce the strategy's primary sticky model for a tenant id."""
    catalog = default_model_catalog()
    ordered = sorted(catalog.values(), key=lambda candidate: candidate.model)
    digest = sha256(tenant_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % len(ordered)
    return ordered[bucket].model


def _bucket(request_id: str) -> float:
    """Reproduce the strategy's explore bucket for a request id."""
    digest = sha256(request_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _strategy(
    *,
    sticky_canary_cost_percent: float = 10.0,
    unavailable: set[str] | None = None,
) -> StickyCanaryCostStrategy:
    """Build a sticky-canary-cost strategy for tests."""
    return StickyCanaryCostStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable or set()),
        sticky_canary_cost_percent=sticky_canary_cost_percent,
    )


def test_sticky_canary_cost_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("sticky-canary-cost") is RoutingStrategyName.STICKY_CANARY_COST


def test_sticky_canary_cost_pins_same_tenant_off_slice() -> None:
    """Off-slice requests sharing tenant_id should stick to the same model."""
    strategy = _strategy(sticky_canary_cost_percent=0.0)
    tenant_id = "tenant-acme"

    first = strategy.choose(_request(metadata={"tenant_id": tenant_id}), _signals())
    second = strategy.choose(_request(metadata={"tenant_id": tenant_id}), _signals())

    assert first.chosen_model == second.chosen_model
    assert first.chosen_model == _primary_model(tenant_id)
    assert "pinned tenant" in first.rationale
    assert first.routing_strategy is RoutingStrategyName.STICKY_CANARY_COST


def test_sticky_canary_cost_explore_blends_toward_cheaper() -> None:
    """Explore-slice traffic should blend toward a cheaper healthy model."""
    catalog = default_model_catalog()
    tenant_id = "tenant-cost-explore"
    sticky_model = _primary_model(tenant_id)
    sticky_cost = catalog[sticky_model].estimate_cost(8, 512)
    cheaper_exists = any(
        candidate.estimate_cost(8, 512) < sticky_cost for candidate in catalog.values()
    )
    assert cheaper_exists, "test requires a sticky primary that is not the cheapest model"

    strategy = _strategy(sticky_canary_cost_percent=100.0)
    decision = strategy.choose(
        _request(request_id="req-explore-cost", metadata={"tenant_id": tenant_id}),
        _signals(),
    )

    assert "explore slice" in decision.rationale
    assert "toward cheaper healthy" in decision.rationale
    assert decision.chosen_model != sticky_model
    assert catalog[decision.chosen_model].estimate_cost(8, 512) < sticky_cost


def test_sticky_canary_cost_explore_falls_back_when_no_cheaper() -> None:
    """Explore slice keeps sticky when no cheaper healthy option exists."""
    catalog = default_model_catalog()
    # Find a tenant whose sticky primary is already the cheapest catalog model.
    cheapest = min(catalog.values(), key=lambda c: (c.estimate_cost(8, 512), c.model))
    tenant_id = None
    for candidate_tenant in (f"tenant-{idx}" for idx in range(200)):
        if _primary_model(candidate_tenant) == cheapest.model:
            tenant_id = candidate_tenant
            break
    assert tenant_id is not None

    strategy = _strategy(sticky_canary_cost_percent=100.0)
    decision = strategy.choose(
        _request(request_id="req-no-cheaper", metadata={"tenant_id": tenant_id}),
        _signals(),
    )

    assert decision.chosen_model == cheapest.model
    assert "no cheaper healthy option" in decision.rationale


def test_sticky_canary_cost_bucket_is_deterministic() -> None:
    """The same request id should always land in the same explore slice."""
    request_id = "req-deterministic-sticky-canary"
    bucket = _bucket(request_id)
    strategy = _strategy(sticky_canary_cost_percent=bucket * 100.0 + 0.01)

    first = strategy.choose(
        _request(request_id=request_id, metadata={"tenant_id": "tenant-det"}),
        _signals(),
    )
    second = strategy.choose(
        _request(request_id=request_id, metadata={"tenant_id": "tenant-det"}),
        _signals(),
    )

    assert first.chosen_model == second.chosen_model
    assert "explore slice" in first.rationale


def test_sticky_canary_cost_rejects_invalid_percent() -> None:
    """Explore percent must stay inside [0.0, 100.0]."""
    with pytest.raises(ValueError, match="sticky_canary_cost_percent"):
        _strategy(sticky_canary_cost_percent=150.0)


def test_sticky_canary_cost_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose sticky-canary-cost."""
    settings = RouterSettings(sticky_canary_cost_percent=12.5)
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
        provider_error_budget_rate=settings.provider_error_budget_rate,
        sticky_canary_cost_percent=settings.sticky_canary_cost_percent,
    )

    strategy = strategies[RoutingStrategyName.STICKY_CANARY_COST]
    assert isinstance(strategy, StickyCanaryCostStrategy)
    assert strategy.strategy_name is RoutingStrategyName.STICKY_CANARY_COST


def test_sticky_canary_cost_settings_default() -> None:
    """RouterSettings expose the sticky canary cost percent default."""
    assert RouterSettings().sticky_canary_cost_percent == 10.0
