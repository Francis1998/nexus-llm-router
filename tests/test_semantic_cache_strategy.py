"""Tests for the semantic-cache routing strategy."""

from router.config import default_model_catalog
from router.model_ids import OPENAI_BALANCED_MODEL
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import CostOptimalStrategy, SemanticCacheStrategy


def _request(*, cache_hit: object | None = None) -> RouterRequest:
    """Build a minimal router request with optional cache metadata."""
    metadata: dict[str, object] = {}
    if cache_hit is not None:
        metadata["cache_hit"] = cache_hit
    return RouterRequest(
        request_id="req-semantic-cache",
        messages=[ChatMessage(content="Hello")],
        metadata=metadata,
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


def test_semantic_cache_hit_prefers_cheapest_eligible_model() -> None:
    """A cache hit should route to the cheapest domain-eligible realtime model."""
    catalog = default_model_catalog()
    strategy = SemanticCacheStrategy(catalog, quality_floor=0.72)
    request, signals = _request(cache_hit=True), _signals()

    decision = strategy.choose(request, signals)

    eligible = [
        candidate
        for candidate in catalog.values()
        if DomainTag.GENERAL in candidate.supports_domains and candidate.supports_realtime
    ]
    cheapest = min(
        eligible,
        key=lambda candidate: (
            candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
            -candidate.quality_score,
            candidate.model,
        ),
    )
    assert decision.chosen_model == cheapest.model
    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert decision.routing_strategy is RoutingStrategyName.SEMANTIC_CACHE
    assert "cache hit" in decision.rationale


def test_semantic_cache_miss_falls_through_to_cost_optimal() -> None:
    """A cache miss should match cost-optimal under the same quality floor."""
    catalog = default_model_catalog()
    quality_floor = 0.72
    strategy = SemanticCacheStrategy(catalog, quality_floor=quality_floor)
    cost_optimal = CostOptimalStrategy(catalog, quality_floor)
    request, signals = _request(cache_hit=False), _signals()

    decision = strategy.choose(request, signals)
    expected = cost_optimal.choose(request, signals)

    assert decision.chosen_model == expected.chosen_model
    assert decision.routing_strategy is RoutingStrategyName.SEMANTIC_CACHE
    assert "cache miss" in decision.rationale


def test_semantic_cache_absent_metadata_treated_as_miss() -> None:
    """Missing cache_hit metadata is a miss and uses cost-optimal selection."""
    catalog = default_model_catalog()
    strategy = SemanticCacheStrategy(catalog, quality_floor=0.72)
    cost_optimal = CostOptimalStrategy(catalog, 0.72)
    request, signals = _request(), _signals()

    decision = strategy.choose(request, signals)

    assert decision.chosen_model == cost_optimal.choose(request, signals).chosen_model
    assert "cache miss" in decision.rationale


def test_semantic_cache_string_true_counts_as_hit() -> None:
    """Stringy truthy cache_hit values from gateways should count as hits."""
    strategy = SemanticCacheStrategy(default_model_catalog(), quality_floor=0.72)

    decision = strategy.choose(_request(cache_hit="true"), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "cache hit" in decision.rationale
