"""Tests for the canary-tier-blend routing strategy."""

from hashlib import sha256

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_FAST_MODEL,
    ANTHROPIC_SAFETY_MODEL,
    OPENAI_BALANCED_MODEL,
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
    CanaryTierBlendStrategy,
    InflightStats,
    LatencyStats,
    ModelTier,
    SuccessStats,
    build_strategies,
    infer_model_tier,
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


def _request(request_id: str = "req-canary-tier") -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(request_id=request_id, messages=[ChatMessage(content="hello")])


def _signals(complexity_score: float = 0.5) -> TaskSignals:
    """Build task signals with a complexity score."""
    return TaskSignals(
        complexity_score=complexity_score,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def _bucket(request_id: str) -> float:
    """Reproduce the strategy's canary bucket for a request id."""
    digest = sha256(request_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _strategy(
    *,
    canary_weight: float,
    unavailable: set[str] | None = None,
    stable_model: str = OPENAI_BALANCED_MODEL,
    canary_model: str = OPENAI_FRONTIER_MODEL,
) -> CanaryTierBlendStrategy:
    """Build a canary-tier-blend strategy for tests."""
    return CanaryTierBlendStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable or set()),
        stable_model=stable_model,
        canary_model=canary_model,
        canary_weight=canary_weight,
    )


def test_canary_tier_blend_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("canary-tier-blend") is RoutingStrategyName.CANARY_TIER_BLEND


def test_canary_tier_blend_prefers_tier_matching_canary_on_slice() -> None:
    """Canary traffic should prefer the canary when it matches the target tier."""
    strategy = _strategy(canary_weight=1.0, canary_model=OPENAI_FRONTIER_MODEL)

    decision = strategy.choose(_request("req-frontier"), _signals(0.9))

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.routing_strategy is RoutingStrategyName.CANARY_TIER_BLEND
    assert infer_model_tier(OPENAI_FRONTIER_MODEL) is ModelTier.FRONTIER
    assert "matches frontier tier" in decision.rationale


def test_canary_tier_blend_still_routes_canary_when_tier_mismatches() -> None:
    """Canary slice traffic should still hit the canary when tiers differ."""
    strategy = _strategy(
        canary_weight=1.0,
        canary_model=ANTHROPIC_FAST_MODEL,
    )

    decision = strategy.choose(_request("req-mismatch"), _signals(0.9))

    assert decision.chosen_model == ANTHROPIC_FAST_MODEL
    assert "tier mismatch" in decision.rationale


def test_canary_tier_blend_off_slice_prefers_matching_tier() -> None:
    """Non-canary traffic should prefer the target tier over raw quality."""
    strategy = _strategy(canary_weight=0.0)

    decision = strategy.choose(_request("req-stable-tier"), _signals(0.9))

    assert infer_model_tier(decision.chosen_model) is ModelTier.FRONTIER
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "frontier tier" in decision.rationale


def test_canary_tier_blend_pauses_unhealthy_canary_for_tier_match() -> None:
    """An unhealthy canary should fall through to tier matching."""
    strategy = _strategy(
        canary_weight=1.0,
        unavailable={"openai"},
        canary_model=OPENAI_FRONTIER_MODEL,
    )

    decision = strategy.choose(_request("req-paused"), _signals(0.9))

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "canary paused" in decision.rationale
    assert "frontier tier" in decision.rationale


def test_canary_tier_blend_falls_back_to_quality_without_tier_match() -> None:
    """When no tier match exists, pick the highest-quality eligible model."""
    catalog = {
        "economy-only": default_model_catalog()[OPENAI_BALANCED_MODEL].model_copy(
            update={"model": "economy-only", "quality_score": 0.95}
        )
    }
    strategy = CanaryTierBlendStrategy(
        catalog,
        _FakeHealth(set()),
        stable_model="economy-only",
        canary_model="economy-only",
        canary_weight=0.0,
    )

    decision = strategy.choose(_request("req-quality"), _signals(0.9))

    assert decision.chosen_model == "economy-only"
    assert "highest-quality" in decision.rationale


def test_canary_tier_blend_rejects_invalid_weight() -> None:
    """Canary weight must stay inside [0.0, 1.0]."""
    with pytest.raises(ValueError, match="canary_weight"):
        _strategy(canary_weight=1.5)


def test_canary_tier_blend_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose canary-tier-blend."""
    catalog = default_model_catalog()
    settings = RouterSettings(canary_weight=0.1)
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

    strategy = strategies[RoutingStrategyName.CANARY_TIER_BLEND]
    assert isinstance(strategy, CanaryTierBlendStrategy)
    assert strategy.strategy_name is RoutingStrategyName.CANARY_TIER_BLEND
