"""Tests for the failover-priority routing strategy."""

import pytest

from router.config import default_model_catalog
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
from router.strategies import FailoverPriorityStrategy
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id="req-failover",
        messages=[ChatMessage(content="Hello")],
    )


def _signals() -> TaskSignals:
    """Build general-domain task signals."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=64,
    )


def _priority() -> list[str]:
    """Return the default ordered preference list."""
    return [
        OPENAI_FRONTIER_MODEL,
        ANTHROPIC_SAFETY_MODEL,
        GEMINI_PRO_MODEL,
        MOONSHOT_BALANCED_MODEL,
    ]


def test_failover_priority_picks_first_when_all_healthy() -> None:
    """With every provider healthy, the first preference wins."""
    strategy = FailoverPriorityStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        _priority(),
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.routing_strategy is RoutingStrategyName.FAILOVER_PRIORITY
    assert "first healthy" in decision.rationale
    assert decision.fallback_chain == [
        ANTHROPIC_SAFETY_MODEL,
        GEMINI_PRO_MODEL,
        MOONSHOT_BALANCED_MODEL,
    ]


def test_failover_priority_skips_unhealthy_provider() -> None:
    """An open circuit on the first preference should advance to the next."""
    health = CircuitBreakerRegistry(failure_threshold=1, recovery_window_seconds=60.0)
    health.record_failure("openai")
    strategy = FailoverPriorityStrategy(default_model_catalog(), health, _priority())

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.chosen_model != OPENAI_FRONTIER_MODEL
    assert OPENAI_FRONTIER_MODEL in decision.fallback_chain


def test_failover_priority_falls_back_when_all_unhealthy() -> None:
    """When every preference is unhealthy, still route to the first listed model."""
    health = CircuitBreakerRegistry(failure_threshold=1, recovery_window_seconds=60.0)
    for provider in ("openai", "anthropic", "google", "moonshot"):
        health.record_failure(provider)
    strategy = FailoverPriorityStrategy(default_model_catalog(), health, _priority())

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert "no healthy preference" in decision.rationale


def test_failover_priority_rejects_empty_list() -> None:
    """An empty preference list should fail fast at construction."""
    with pytest.raises(ValueError, match="at least one model"):
        FailoverPriorityStrategy(default_model_catalog(), CircuitBreakerRegistry(), [])


def test_failover_priority_rejects_unknown_models() -> None:
    """Priority entries must resolve to at least one catalog model."""
    with pytest.raises(ValueError, match="not in catalog"):
        FailoverPriorityStrategy(
            default_model_catalog(),
            CircuitBreakerRegistry(),
            ["not-a-real-model"],
        )
