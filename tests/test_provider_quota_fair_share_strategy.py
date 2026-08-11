"""Tests for the provider-quota-fair-share routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_PRO_MODEL,
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
    ProviderQuotaFairShareStrategy,
    ProviderRequestShareStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(request_id: str = "req-provider-quota") -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id=request_id,
        messages=[ChatMessage(content="Route this request fairly.")],
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _strategy(lookback: int = 100) -> ProviderQuotaFairShareStrategy:
    """Build provider-quota-fair-share with a fresh rolling window."""
    return ProviderQuotaFairShareStrategy(
        default_model_catalog(),
        ProviderRequestShareStats(lookback),
    )


def test_provider_quota_fair_share_enum_parses() -> None:
    """The API header parser can resolve the new strategy value."""
    assert (
        RoutingStrategyName("provider-quota-fair-share")
        is RoutingStrategyName.PROVIDER_QUOTA_FAIR_SHARE
    )


def test_provider_request_share_stats_rejects_invalid_lookback() -> None:
    """The rolling request window must retain at least one observation."""
    with pytest.raises(ValueError, match="lookback must be >= 1"):
        ProviderRequestShareStats(0)


def test_provider_request_share_stats_evicts_old_observations() -> None:
    """Only the configured number of recent provider selections count."""
    stats = ProviderRequestShareStats(lookback=2)
    stats.observe("anthropic")
    stats.observe("openai")
    stats.observe("openai")

    providers = {"anthropic", "openai"}
    assert stats.observation_count(providers) == 2
    assert stats.request_share("anthropic", providers) == 0.0
    assert stats.request_share("openai", providers) == 1.0


def test_provider_quota_fair_share_cold_start_picks_top_quality() -> None:
    """An empty share window should retain quality-first routing."""
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_QUOTA_FAIR_SHARE
    assert "cold start" in decision.rationale
    assert "equal target 25.00%" in decision.rationale


def test_provider_quota_fair_share_rotates_under_share_providers() -> None:
    """Recent selections should shed overloaded providers until shares equalize."""
    strategy = _strategy(lookback=4)

    decisions = [
        strategy.choose(_request(f"req-{index}"), _signals()).chosen_model for index in range(1, 6)
    ]

    assert decisions == [
        ANTHROPIC_SAFETY_MODEL,
        OPENAI_FRONTIER_MODEL,
        GEMINI_PRO_MODEL,
        MOONSHOT_BALANCED_MODEL,
        ANTHROPIC_SAFETY_MODEL,
    ]


def test_provider_quota_fair_share_uses_current_domain_provider_set() -> None:
    """Fair share should be recomputed from providers eligible for the request domain."""
    strategy = _strategy()

    first = strategy.choose(_request("req-medical-1"), _signals(DomainTag.MEDICAL))
    second = strategy.choose(_request("req-medical-2"), _signals(DomainTag.MEDICAL))

    assert first.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert second.chosen_model == GEMINI_PRO_MODEL
    assert "equal share 50.00%" in second.rationale


def test_provider_quota_fair_share_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose provider-quota-fair-share."""
    settings = RouterSettings(provider_quota_lookback=7)
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
        provider_quota_lookback=settings.provider_quota_lookback,
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_QUOTA_FAIR_SHARE]
    assert isinstance(strategy, ProviderQuotaFairShareStrategy)


def test_provider_quota_fair_share_settings_default() -> None:
    """RouterSettings expose a bounded default fair-share window."""
    assert RouterSettings().provider_quota_lookback == 100
