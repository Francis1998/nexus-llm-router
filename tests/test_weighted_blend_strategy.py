"""Tests for the weighted-blend routing strategy."""

import pytest

from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_BALANCED_MODEL,
)
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import LatencyStats, WeightedBlendStrategy


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


def test_weighted_blend_pure_quality_picks_top_quality_model() -> None:
    """With all weight on quality, the highest-quality model is chosen."""
    strategy = WeightedBlendStrategy(
        default_model_catalog(),
        LatencyStats(),
        quality_weight=1.0,
        cost_weight=0.0,
        latency_weight=0.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy == RoutingStrategyName.WEIGHTED_BLEND


def test_weighted_blend_pure_cost_picks_cheapest_model() -> None:
    """With all weight on cost, the cheapest general model is chosen."""
    strategy = WeightedBlendStrategy(
        default_model_catalog(),
        LatencyStats(),
        quality_weight=0.0,
        cost_weight=1.0,
        latency_weight=0.0,
    )

    decision = strategy.choose(_request(), _signals())

    # openai balanced ($0.0002/$0.0008) is the cheapest general candidate.
    assert decision.chosen_model == OPENAI_BALANCED_MODEL


def test_weighted_blend_zero_weights_fall_back_to_quality() -> None:
    """All-zero weights degrade to a pure-quality selection instead of failing."""
    strategy = WeightedBlendStrategy(
        default_model_catalog(),
        LatencyStats(),
        quality_weight=0.0,
        cost_weight=0.0,
        latency_weight=0.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_weighted_blend_rejects_negative_weight() -> None:
    """A negative weight fails fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        WeightedBlendStrategy(
            default_model_catalog(),
            LatencyStats(),
            quality_weight=0.5,
            cost_weight=-0.1,
            latency_weight=0.2,
        )


def test_weighted_blend_latency_weight_prefers_faster_provider() -> None:
    """Latency weighting steers selection toward the lower-p95 provider."""
    catalog = default_model_catalog()
    latency_stats = LatencyStats()
    # Make moonshot look very fast and openai/anthropic slow.
    for _ in range(5):
        latency_stats.observe("anthropic", 4000.0)
        latency_stats.observe("openai", 4000.0)
        latency_stats.observe("google", 4000.0)
        latency_stats.observe("moonshot", 10.0)
    strategy = WeightedBlendStrategy(
        catalog,
        latency_stats,
        quality_weight=0.0,
        cost_weight=0.0,
        latency_weight=1.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
