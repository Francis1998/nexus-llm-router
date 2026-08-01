"""Tests for the canary-cost-blend routing strategy."""

from hashlib import sha256

import pytest

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
    CanaryCostBlendStrategy,
    InflightStats,
    LatencyStats,
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


def _request(request_id: str = "req-canary-cost") -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(request_id=request_id, messages=[ChatMessage(content="hello")])


def _signals() -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def _bucket(request_id: str) -> float:
    """Reproduce the strategy's explore bucket for a request id."""
    digest = sha256(request_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _strategy(
    *,
    canary_cost_blend_percent: float,
    unavailable: set[str] | None = None,
) -> CanaryCostBlendStrategy:
    """Build a canary-cost-blend strategy for tests."""
    return CanaryCostBlendStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable or set()),
        canary_cost_blend_percent=canary_cost_blend_percent,
    )


def test_canary_cost_blend_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("canary-cost-blend") is RoutingStrategyName.CANARY_COST_BLEND


def test_canary_cost_blend_picks_cheapest_healthy_off_slice() -> None:
    """Default traffic should pick the cheapest healthy model."""
    strategy = _strategy(canary_cost_blend_percent=0.0)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert decision.routing_strategy is RoutingStrategyName.CANARY_COST_BLEND
    assert "cheapest healthy model" in decision.rationale


def test_canary_cost_blend_explores_next_cheaper_tier_on_slice() -> None:
    """Explore slice traffic should step to the next-cheaper healthy model."""
    strategy = _strategy(canary_cost_blend_percent=100.0)

    decision = strategy.choose(_request("req-explore-cost"), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "next-cheaper healthy tier" in decision.rationale
    assert "explore slice" in decision.rationale


def test_canary_cost_blend_skips_unhealthy_when_picking_cheapest() -> None:
    """Unhealthy providers should be excluded from the cheapest pool."""
    strategy = _strategy(canary_cost_blend_percent=0.0, unavailable={"openai"})

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "cheapest healthy model" in decision.rationale


def test_canary_cost_blend_explore_falls_back_with_single_healthy_option() -> None:
    """Explore slice with one healthy model should still return that model."""
    catalog = {
        "solo-model": default_model_catalog()[OPENAI_BALANCED_MODEL].model_copy(
            update={"model": "solo-model"}
        )
    }
    strategy = CanaryCostBlendStrategy(
        catalog,
        _FakeHealth(set()),
        canary_cost_blend_percent=100.0,
    )

    decision = strategy.choose(_request("req-single-healthy"), _signals())

    assert decision.chosen_model == "solo-model"
    assert "only one healthy tier" in decision.rationale


def test_canary_cost_blend_bucket_is_deterministic() -> None:
    """The same request id should always land in the same explore slice."""
    request_id = "req-deterministic-cost"
    bucket = _bucket(request_id)
    strategy = _strategy(canary_cost_blend_percent=bucket * 100.0 + 0.01)

    first = strategy.choose(_request(request_id), _signals())
    second = strategy.choose(_request(request_id), _signals())

    assert first.chosen_model == second.chosen_model
    assert "explore slice" in first.rationale


def test_canary_cost_blend_rejects_invalid_percent() -> None:
    """Explore percent must stay inside [0.0, 100.0]."""
    with pytest.raises(ValueError, match="canary_cost_blend_percent"):
        _strategy(canary_cost_blend_percent=150.0)


def test_canary_cost_blend_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose canary-cost-blend."""
    catalog = default_model_catalog()
    settings = RouterSettings(canary_cost_blend_percent=10.0)
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
        canary_cost_blend_percent=settings.canary_cost_blend_percent,
    )

    strategy = strategies[RoutingStrategyName.CANARY_COST_BLEND]
    assert isinstance(strategy, CanaryCostBlendStrategy)
    assert strategy.strategy_name is RoutingStrategyName.CANARY_COST_BLEND
