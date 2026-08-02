"""Tests for the sticky-tenant-hash routing strategy."""

from hashlib import sha256

from router.config import RouterSettings, default_model_catalog
from router.model_ids import MOONSHOT_BALANCED_MODEL, OPENAI_BALANCED_MODEL
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
    StickyTenantHashStrategy,
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
    request_id: str = "req-sticky-tenant",
    metadata: dict[str, str] | None = None,
    user_id: str = "anonymous",
    session_id: str = "default",
) -> RouterRequest:
    """Build a router request for sticky-tenant tests."""
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


def _strategy(unavailable: set[str] | None = None) -> StickyTenantHashStrategy:
    """Build a sticky-tenant-hash strategy for tests."""
    return StickyTenantHashStrategy(default_model_catalog(), _FakeHealth(unavailable or set()))


def test_sticky_tenant_hash_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("sticky-tenant-hash") is RoutingStrategyName.STICKY_TENANT_HASH


def test_sticky_tenant_hash_pins_same_tenant_to_same_model() -> None:
    """Requests sharing metadata tenant_id should route to the same model."""
    strategy = _strategy()
    tenant_id = "tenant-acme"

    first = strategy.choose(_request(metadata={"tenant_id": tenant_id}), _signals())
    second = strategy.choose(_request(metadata={"tenant_id": tenant_id}), _signals())

    assert first.chosen_model == second.chosen_model
    assert first.chosen_model == _primary_model(tenant_id)
    assert "sticky-tenant-hash pinned tenant" in first.rationale


def test_sticky_tenant_hash_prefers_metadata_tenant_over_session() -> None:
    """metadata.tenant_id should win over session_id for stickiness."""
    strategy = _strategy()
    tenant_id = "tenant-priority"

    decision = strategy.choose(
        _request(metadata={"tenant_id": tenant_id}, session_id="other-session"),
        _signals(),
    )

    assert decision.chosen_model == _primary_model(tenant_id)
    assert f"tenant '{tenant_id}'" in decision.rationale


def test_sticky_tenant_hash_falls_back_to_user_id_without_metadata() -> None:
    """user_id should be used when metadata omits tenant identifiers."""
    strategy = _strategy()
    user_id = "user-42"

    decision = strategy.choose(_request(user_id=user_id), _signals())

    assert decision.chosen_model == _primary_model(user_id)
    assert f"tenant '{user_id}'" in decision.rationale


def test_sticky_tenant_hash_failovers_when_primary_unhealthy() -> None:
    """An unhealthy sticky primary should advance to the next healthy ring slot."""
    tenant_id = "tenant-failover"
    primary = _primary_model(tenant_id)
    primary_provider = default_model_catalog()[primary].provider
    strategy = _strategy(unavailable={primary_provider})

    decision = strategy.choose(_request(metadata={"tenant_id": tenant_id}), _signals())

    assert decision.chosen_model != primary
    assert "failover offset" in decision.rationale


def test_sticky_tenant_hash_differs_from_session_only_sticky_session() -> None:
    """Tenant hashing should diverge from session-only sticky-session mapping."""
    from router.strategies import StickySessionStrategy

    tenant_strategy = _strategy()
    session_strategy = StickySessionStrategy(default_model_catalog())
    request = _request(metadata={"tenant_id": "tenant-unique"}, session_id="session-unique")

    tenant_decision = tenant_strategy.choose(request, _signals())
    session_decision = session_strategy.choose(request, _signals())

    assert tenant_decision.chosen_model == _primary_model("tenant-unique")
    assert tenant_decision.chosen_model != session_decision.chosen_model


def test_sticky_tenant_hash_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose sticky-tenant-hash."""
    catalog = default_model_catalog()
    settings = RouterSettings()
    strategies = build_strategies(
        catalog,
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
    )

    strategy = strategies[RoutingStrategyName.STICKY_TENANT_HASH]
    assert isinstance(strategy, StickyTenantHashStrategy)
    assert strategy.strategy_name is RoutingStrategyName.STICKY_TENANT_HASH


def test_sticky_tenant_hash_distinct_tenants_can_pick_different_models() -> None:
    """Different tenant ids should be able to land on different primaries."""
    strategy = _strategy()
    models = {
        strategy.choose(
            _request(metadata={"tenant_id": f"tenant-{index}"}),
            _signals(),
        ).chosen_model
        for index in range(12)
    }

    assert OPENAI_BALANCED_MODEL in models or MOONSHOT_BALANCED_MODEL in models
    assert len(models) > 1
