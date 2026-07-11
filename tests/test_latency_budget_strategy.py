"""Tests for the latency-budget routing strategy."""

import pytest

from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_PRO_MODEL,
    MOONSHOT_BALANCED_MODEL,
)
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import LatencyBudgetStrategy, LatencyStats


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(request_id="req-1", messages=[ChatMessage(content="hello")])


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_latency_budget_cold_start_picks_top_quality_model() -> None:
    """With no latency observed yet, every provider is within SLA (best quality)."""
    strategy = LatencyBudgetStrategy(default_model_catalog(), LatencyStats(), latency_sla_ms=750.0)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy == RoutingStrategyName.LATENCY_BUDGET
    assert "SLA" in decision.rationale


def test_latency_budget_excludes_slow_high_quality_providers() -> None:
    """When top-quality providers breach the SLA, the best fast model wins.

    OpenAI and Anthropic host the highest-quality models but here report p95
    latencies above the 500ms SLA, while Google (~200ms) and Moonshot (~400ms)
    stay within it. Among the within-SLA candidates the Gemini Pro model has the
    highest quality (0.95), so quality is traded for speed only as far as the SLA
    requires.
    """
    latency_stats = LatencyStats()
    latency_stats.observe("openai", 800.0)
    latency_stats.observe("anthropic", 900.0)
    latency_stats.observe("google", 200.0)
    latency_stats.observe("moonshot", 400.0)
    strategy = LatencyBudgetStrategy(default_model_catalog(), latency_stats, latency_sla_ms=500.0)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "within 500ms SLA" in decision.rationale


def test_latency_budget_unreachable_sla_falls_back_to_fastest() -> None:
    """When no provider meets the SLA, the lowest-p95 model is chosen."""
    latency_stats = LatencyStats()
    latency_stats.observe("openai", 800.0)
    latency_stats.observe("anthropic", 900.0)
    latency_stats.observe("google", 700.0)
    latency_stats.observe("moonshot", 600.0)
    strategy = LatencyBudgetStrategy(default_model_catalog(), latency_stats, latency_sla_ms=100.0)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "no provider within" in decision.rationale


def test_latency_budget_respects_domain_support() -> None:
    """Only medical-capable models are considered for a medical prompt."""
    strategy = LatencyBudgetStrategy(default_model_catalog(), LatencyStats(), latency_sla_ms=750.0)

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    candidate = default_model_catalog()[decision.chosen_model]
    assert DomainTag.MEDICAL in candidate.supports_domains
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_latency_budget_rejects_negative_sla() -> None:
    """A negative latency SLA fails fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        LatencyBudgetStrategy(default_model_catalog(), LatencyStats(), latency_sla_ms=-1.0)
