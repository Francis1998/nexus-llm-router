"""Tests for the provider-error-budget-shed routing strategy."""

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
    ProviderErrorBudgetShedStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(max_tokens: int = 512) -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id="req-provider-error-budget",
        messages=[ChatMessage(content="Analyze provider reliability.")],
        max_tokens=max_tokens,
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


def _strategy(
    *,
    provider_health: CircuitBreakerRegistry | None = None,
    success_stats: SuccessStats | None = None,
    provider_error_budget_rate: float = 0.15,
) -> ProviderErrorBudgetShedStrategy:
    """Build provider-error-budget-shed with overridable dependencies."""
    return ProviderErrorBudgetShedStrategy(
        default_model_catalog(),
        provider_health or CircuitBreakerRegistry(),
        success_stats or SuccessStats(),
        provider_error_budget_rate=provider_error_budget_rate,
    )


def test_provider_error_budget_shed_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("provider-error-budget-shed")
        is RoutingStrategyName.PROVIDER_ERROR_BUDGET_SHED
    )


def test_provider_error_budget_shed_cold_start_picks_top_quality() -> None:
    """Cold SuccessStats treat every provider as 0% error."""
    strategy = _strategy()

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_ERROR_BUDGET_SHED
    assert "under error budget 15.00%" in decision.rationale
    assert "anthropic error 0.00%" in decision.rationale


def test_provider_error_budget_shed_skips_over_budget_top_provider() -> None:
    """An over-budget top provider should lose to the best under-budget model."""
    stats = SuccessStats()
    stats.observe("anthropic", success=False)
    stats.observe("openai", success=True)
    strategy = _strategy(success_stats=stats)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "openai error 0.00%" in decision.rationale
    assert ANTHROPIC_SAFETY_MODEL not in decision.fallback_chain[:1]


def test_provider_error_budget_shed_all_over_budget_uses_lowest_error_then_quality() -> None:
    """When every provider is over budget, lowest error wins before quality."""
    stats = SuccessStats()
    for provider in ("openai", "google"):
        for _ in range(3):
            stats.observe(provider, success=True)
        for _ in range(2):
            stats.observe(provider, success=False)
    stats.observe("anthropic", success=True)
    stats.observe("anthropic", success=False)
    stats.observe("moonshot", success=False)
    strategy = _strategy(success_stats=stats)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert "every eligible provider over error budget 15.00%" in decision.rationale
    assert "openai (40.00%)" in decision.rationale


def test_provider_error_budget_shed_prefers_closed_circuit_when_available() -> None:
    """Circuit-open providers should not win while any healthy candidate remains."""
    health = CircuitBreakerRegistry(failure_threshold=1, recovery_window_seconds=60.0)
    health.record_failure("anthropic")
    strategy = _strategy(provider_health=health)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert ANTHROPIC_SAFETY_MODEL not in decision.fallback_chain[:1]


def test_provider_error_budget_shed_rejects_invalid_budget_rate() -> None:
    """Invalid error-budget rates fail fast at construction."""
    with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
        _strategy(provider_error_budget_rate=1.1)


def test_provider_error_budget_shed_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose provider-error-budget-shed."""
    settings = RouterSettings(provider_error_budget_rate=0.2)
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
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_ERROR_BUDGET_SHED]
    assert isinstance(strategy, ProviderErrorBudgetShedStrategy)
    assert strategy.strategy_name is RoutingStrategyName.PROVIDER_ERROR_BUDGET_SHED
