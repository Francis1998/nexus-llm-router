"""Tests for provider-canary-shadow-split routing."""

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
    CanaryShadowSplitStats,
    CanaryShadowSplitStrategy,
    InflightStats,
    LatencyStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _request(request_id: str = "req-canary-shadow", tenant: str | None = None) -> RouterRequest:
    metadata = {} if tenant is None else {"tenant_id": tenant}
    return RouterRequest(
        request_id=request_id,
        messages=[ChatMessage(content="Compare provider quality safely.")],
        metadata=metadata,
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.6,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _strategy(
    stats: CanaryShadowSplitStats | None = None,
    *,
    unavailable: set[str] | None = None,
    preferred_provider: str = "openai",
    shadow_percent: float = 100.0,
) -> CanaryShadowSplitStrategy:
    return CanaryShadowSplitStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        stats or CanaryShadowSplitStats(),
        preferred_provider=preferred_provider,
        shadow_percent=shadow_percent,
    )


def test_provider_canary_shadow_split_enum_parses() -> None:
    assert (
        RoutingStrategyName("provider-canary-shadow-split")
        is RoutingStrategyName.PROVIDER_CANARY_SHADOW_SPLIT
    )


def test_provider_canary_shadow_split_keeps_preferred_provider_primary() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_CANARY_SHADOW_SPLIT


def test_provider_canary_shadow_split_selects_different_provider_shadow() -> None:
    decision = _strategy(shadow_percent=100.0).choose(_request(), _signals())

    assert ANTHROPIC_SAFETY_MODEL in decision.rationale
    assert "shadow candidate" in decision.rationale
    assert "queued for comparison" in decision.rationale
    assert decision.fallback_chain[0] == ANTHROPIC_SAFETY_MODEL


def test_provider_canary_shadow_split_off_slice_keeps_primary_only() -> None:
    stats = CanaryShadowSplitStats()
    decision = _strategy(stats, shadow_percent=0.0).choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert "held off" in decision.rationale
    assert stats.total_shadows == 0


def test_provider_canary_shadow_split_stats_count_provider_pairs() -> None:
    stats = CanaryShadowSplitStats()
    _strategy(stats).choose(_request(), _signals())

    assert stats.primary_count("openai") == 1
    assert stats.shadow_count("anthropic") == 1
    assert stats.split_count("openai", "anthropic") == 1
    assert stats.total_shadows == 1


def test_provider_canary_shadow_split_tenant_cohort_is_deterministic() -> None:
    strategy = _strategy(shadow_percent=50.0)
    first = strategy._shadow_bucket(_request("request-a", tenant="acme"))  # noqa: SLF001
    second = strategy._shadow_bucket(_request("request-b", tenant="acme"))  # noqa: SLF001

    assert first == second


def test_provider_canary_shadow_split_falls_back_when_preferred_is_unhealthy() -> None:
    decision = _strategy(unavailable={"openai"}).choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.provider == "anthropic"
    assert "preferred provider 'openai' unavailable" in decision.rationale


def test_provider_canary_shadow_split_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="preferred_provider must not be empty"):
        _strategy(preferred_provider=" ")
    with pytest.raises(ValueError, match="shadow_percent must be within"):
        _strategy(shadow_percent=100.1)


def test_provider_canary_shadow_split_registered_by_strategy_factory() -> None:
    settings = RouterSettings(
        provider_canary_primary_provider="anthropic",
        provider_canary_shadow_percent=17.0,
    )
    stats = CanaryShadowSplitStats()
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
        provider_canary_shadow_stats=stats,
        provider_canary_primary_provider=settings.provider_canary_primary_provider,
        provider_canary_shadow_percent=settings.provider_canary_shadow_percent,
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_CANARY_SHADOW_SPLIT]
    assert isinstance(strategy, CanaryShadowSplitStrategy)
    assert strategy._shadow_stats is stats  # noqa: SLF001
    assert strategy._preferred_provider == "anthropic"  # noqa: SLF001
    assert strategy._shadow_percent == 17.0  # noqa: SLF001
    assert RouterSettings().provider_canary_primary_provider == "openai"
    assert RouterSettings().provider_canary_shadow_percent == 5.0
