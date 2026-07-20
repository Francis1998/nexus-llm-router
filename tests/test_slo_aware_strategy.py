"""Tests for the SLO-aware routing strategy."""

import pytest

from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
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
from router.strategies import SloAwareStrategy, SuccessStats


def _request() -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id="req-slo",
        messages=[ChatMessage(content="Hello")],
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for a domain."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=64,
    )


def test_slo_aware_picks_top_quality_when_all_providers_healthy() -> None:
    """With no failure observations, every provider meets the SLO."""
    strategy = SloAwareStrategy(default_model_catalog(), SuccessStats(), availability_slo=0.99)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.SLO_AWARE
    assert "slo-aware" in decision.rationale


def test_slo_aware_excludes_providers_below_availability_slo() -> None:
    """A degraded high-quality provider is skipped for one that meets the SLO.

    Anthropic (Claude Sonnet 4.6) is the catalog quality leader, but with a
    50% success rate it fails a 99% availability SLO. OpenAI stays healthy, so
    GPT-5.5 wins among SLO-compliant candidates.
    """
    stats = SuccessStats()
    stats.observe("anthropic", success=True)
    stats.observe("anthropic", success=False)
    stats.observe("openai", success=True)
    stats.observe("openai", success=True)
    strategy = SloAwareStrategy(default_model_catalog(), stats, availability_slo=0.99)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.chosen_model != ANTHROPIC_SAFETY_MODEL


def test_slo_aware_falls_back_to_highest_success_rate_when_none_meet_slo() -> None:
    """When every provider is below the SLO, pick the highest success rate."""
    stats = SuccessStats()
    for _ in range(9):
        stats.observe("anthropic", success=True)
    stats.observe("anthropic", success=False)  # 90%
    for _ in range(5):
        stats.observe("openai", success=True)
        stats.observe("openai", success=False)  # 50%
    for _ in range(8):
        stats.observe("google", success=False)
    stats.observe("google", success=True)
    stats.observe("moonshot", success=False)
    strategy = SloAwareStrategy(default_model_catalog(), stats, availability_slo=0.99)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "no provider meeting" in decision.rationale


def test_slo_aware_respects_domain_eligibility() -> None:
    """Only domain-eligible models may be chosen for a specialized domain."""
    strategy = SloAwareStrategy(default_model_catalog(), SuccessStats(), availability_slo=0.99)

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert DomainTag.MEDICAL in default_model_catalog()[decision.chosen_model].supports_domains


def test_slo_aware_rejects_out_of_range_slo() -> None:
    """Availability SLO outside [0.0, 1.0] should fail fast at construction."""
    with pytest.raises(ValueError, match="availability_slo must be within"):
        SloAwareStrategy(default_model_catalog(), SuccessStats(), availability_slo=1.5)


def test_success_stats_cold_start_reports_full_health() -> None:
    """Providers with no observations are treated as fully healthy."""
    assert SuccessStats().success_rate("openai") == 1.0
