"""Tests for the retry-budget-aware-failover routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, GEMINI_PRO_MODEL
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
    RetryBudgetAwareFailoverStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(retry_remaining: int | None = None) -> RouterRequest:
    """Build a router request with optional retry_remaining metadata."""
    metadata = {} if retry_remaining is None else {"retry_remaining": retry_remaining}
    return RouterRequest(
        request_id="req-retry-budget",
        messages=[ChatMessage(content="hello")],
        metadata=metadata,
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_retry_budget_aware_failover_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("retry-budget-aware-failover")
        is RoutingStrategyName.RETRY_BUDGET_AWARE_FAILOVER
    )


def test_retry_budget_aware_failover_prefers_quality_with_budget() -> None:
    """With retries remaining, prefer highest-quality healthy model."""
    strategy = RetryBudgetAwareFailoverStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        LatencyStats(),
        retry_budget_default=3,
    )

    decision = strategy.choose(_request(3), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "highest-quality healthy" in decision.rationale


def test_retry_budget_aware_failover_goes_low_latency_when_budget_low() -> None:
    """Near-exhausted retry budget failovers to lowest latency healthy model."""
    latency_stats = LatencyStats()
    latency_stats.observe("openai", 900.0)
    latency_stats.observe("anthropic", 1200.0)
    latency_stats.observe("google", 300.0)
    latency_stats.observe("moonshot", 500.0)
    strategy = RetryBudgetAwareFailoverStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        latency_stats,
        retry_budget_default=3,
    )

    decision = strategy.choose(_request(1), _signals())

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "failover to lowest-latency" in decision.rationale


def test_retry_budget_aware_failover_uses_default_when_metadata_missing() -> None:
    """Missing metadata uses NEXUS_RETRY_BUDGET_DEFAULT."""
    strategy = RetryBudgetAwareFailoverStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        LatencyStats(),
        retry_budget_default=3,
    )

    decision = strategy.choose(_request(None), _signals())

    assert "remaining retries 3" in decision.rationale
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_retry_budget_aware_failover_rejects_negative_default() -> None:
    """A negative default retry budget fails fast at construction."""
    with pytest.raises(ValueError, match=">= 0"):
        RetryBudgetAwareFailoverStrategy(
            default_model_catalog(),
            CircuitBreakerRegistry(),
            LatencyStats(),
            retry_budget_default=-1,
        )


def test_retry_budget_aware_failover_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose retry-budget-aware-failover."""
    catalog = default_model_catalog()
    settings = RouterSettings(retry_budget_default=2)
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
        retry_budget_default=settings.retry_budget_default,
    )

    strategy = strategies[RoutingStrategyName.RETRY_BUDGET_AWARE_FAILOVER]
    assert isinstance(strategy, RetryBudgetAwareFailoverStrategy)
    assert strategy.strategy_name is RoutingStrategyName.RETRY_BUDGET_AWARE_FAILOVER
