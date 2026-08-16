"""Tests for provider-error-budget-reset routing."""

import time

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
    ProviderErrorBudgetResetStats,
    ProviderErrorBudgetResetStrategy,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    return RouterRequest(
        request_id="req-error-reset",
        messages=[ChatMessage(content="Route around timed provider errors")],
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=100,
    )


def _strategy(
    stats: ProviderErrorBudgetResetStats | None = None,
    *,
    fraction: float = 0.15,
) -> ProviderErrorBudgetResetStrategy:
    return ProviderErrorBudgetResetStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        stats or ProviderErrorBudgetResetStats(),
        error_budget_fraction=fraction,
    )


def test_provider_error_budget_reset_enum_parses() -> None:
    assert (
        RoutingStrategyName("provider-error-budget-reset")
        is RoutingStrategyName.PROVIDER_ERROR_BUDGET_RESET
    )


def test_provider_error_budget_reset_cold_start_prefers_quality() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_ERROR_BUDGET_RESET
    assert "within 15.00% timed error budget" in decision.rationale


def test_provider_error_budget_reset_temporarily_sheds_over_budget_provider() -> None:
    stats = ProviderErrorBudgetResetStats()
    stats.observe("anthropic", success=False)

    decision = _strategy(stats).choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert ANTHROPIC_SAFETY_MODEL not in decision.fallback_chain[:1]


def test_provider_error_budget_reset_restores_provider_after_timer() -> None:
    stats = ProviderErrorBudgetResetStats(reset_seconds=0.1)
    stats.observe("anthropic", success=False, now=time.monotonic() - 1.0)

    decision = _strategy(stats).choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert stats.error_rate("anthropic") == 0.0
    assert "reset every 0.1s" in decision.rationale


def test_provider_error_budget_reset_all_over_budget_uses_soonest_reset() -> None:
    stats = ProviderErrorBudgetResetStats(reset_seconds=60.0)
    now = time.monotonic()
    stats.observe("anthropic", success=False, now=now - 50.0)
    for provider in ("openai", "google", "moonshot"):
        stats.observe(provider, success=False, now=now - 5.0)

    decision = _strategy(stats).choose(_request(), _signals())

    assert decision.provider == "anthropic"
    assert "every healthy provider temporarily shed" in decision.rationale


def test_provider_error_budget_reset_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
        _strategy(fraction=1.1)


def test_provider_error_budget_reset_registered_by_strategy_factory() -> None:
    settings = RouterSettings(
        provider_error_budget_reset_fraction=0.25,
        provider_error_budget_reset_seconds=90.0,
    )
    stats = ProviderErrorBudgetResetStats(settings.provider_error_budget_reset_seconds)
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
        provider_error_budget_reset_fraction=settings.provider_error_budget_reset_fraction,
        provider_error_budget_reset_seconds=settings.provider_error_budget_reset_seconds,
        provider_error_budget_reset_stats=stats,
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_ERROR_BUDGET_RESET]
    assert isinstance(strategy, ProviderErrorBudgetResetStrategy)
    assert strategy._error_budget_stats is stats  # noqa: SLF001
    assert strategy._error_budget_fraction == 0.25  # noqa: SLF001
    assert RouterSettings().provider_error_budget_reset_seconds == 60.0
