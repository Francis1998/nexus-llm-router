"""Tests for the sticky-region-failover routing strategy."""

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    MOONSHOT_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
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
    StickyRegionFailoverStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    """Deterministic provider-health view for unit tests."""

    def __init__(self, unavailable: set[str]) -> None:
        """Store providers considered unavailable.

        Args:
            unavailable: Provider names whose circuits are open.
        """
        self._unavailable = unavailable

    def is_available(self, provider: str) -> bool:
        """Return whether a provider is routable."""
        return provider not in self._unavailable


def _request(
    *,
    session_id: str = "session-1",
    region: str | None = None,
) -> RouterRequest:
    """Build a router request with session and optional region."""
    return RouterRequest(
        request_id="req-sticky-region",
        session_id=session_id,
        messages=[ChatMessage(content="Hello")],
        region=region,
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=64,
    )


def test_sticky_region_failover_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("sticky-region-failover") is RoutingStrategyName.STICKY_REGION_FAILOVER
    )


def test_sticky_region_failover_prefers_requested_region() -> None:
    """A CN request should stick to a CN-capable model in that region pool."""
    strategy = StickyRegionFailoverStrategy(
        default_model_catalog(),
        _FakeHealth(set()),
        region_preferences=["eu", "us", "cn", "global"],
    )

    decision = strategy.choose(_request(region="cn"), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert decision.routing_strategy is RoutingStrategyName.STICKY_REGION_FAILOVER
    assert "cn" in decision.rationale


def test_sticky_region_failover_pins_session_within_region_pool() -> None:
    """The same session id must map to the same model inside a region pool."""
    strategy = StickyRegionFailoverStrategy(
        default_model_catalog(),
        _FakeHealth(set()),
        region_preferences=["eu", "us", "global"],
    )

    first = strategy.choose(_request(session_id="sticky-eu", region="eu"), _signals())
    second = strategy.choose(_request(session_id="sticky-eu", region="eu"), _signals())

    assert first.chosen_model == second.chosen_model
    assert "pinned session 'sticky-eu'" in first.rationale


def test_sticky_region_failover_advances_when_preferred_region_unhealthy() -> None:
    """An unhealthy EU pool should failover to the next healthy region preference."""
    strategy = StickyRegionFailoverStrategy(
        default_model_catalog(),
        _FakeHealth({"anthropic", "google"}),
        region_preferences=["eu", "us", "global"],
    )

    decision = strategy.choose(_request(region="eu"), _signals())

    assert decision.provider == "openai"
    assert decision.chosen_model in {OPENAI_FRONTIER_MODEL, "gpt-4.1-mini"}
    assert "us" in decision.rationale or "fallback" in decision.rationale


def test_sticky_region_failover_request_region_is_first_preference() -> None:
    """The request region should be tried before configured failover preferences."""
    strategy = StickyRegionFailoverStrategy(
        default_model_catalog(),
        _FakeHealth(set()),
        region_preferences=["us", "global"],
    )

    decision = strategy.choose(_request(region="eu"), _signals())

    candidate = default_model_catalog()[decision.chosen_model]
    assert "eu" in {region.lower() for region in candidate.supported_regions}
    assert "eu" in decision.rationale


def test_sticky_region_failover_respects_domain_eligibility() -> None:
    """Region failover still requires domain support."""
    strategy = StickyRegionFailoverStrategy(
        default_model_catalog(),
        _FakeHealth(set()),
        region_preferences=["eu", "global"],
    )

    decision = strategy.choose(_request(region="eu"), _signals(DomainTag.MEDICAL))

    candidate = default_model_catalog()[decision.chosen_model]
    assert DomainTag.MEDICAL in candidate.supports_domains


def test_sticky_region_failover_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose sticky-region-failover."""
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
        sticky_region_failover_preferences=settings.sticky_region_failover_preferences,
    )

    strategy = strategies[RoutingStrategyName.STICKY_REGION_FAILOVER]
    assert isinstance(strategy, StickyRegionFailoverStrategy)
    assert strategy.strategy_name is RoutingStrategyName.STICKY_REGION_FAILOVER
