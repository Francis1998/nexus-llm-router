"""Tests for the complexity-tier routing strategy."""

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
from router.strategies import ComplexityTierStrategy


def _request(max_tokens: int = 256) -> RouterRequest:
    """Build a minimal router request for complexity-tier tests.

    Args:
        max_tokens: Maximum output tokens for the request.

    Returns:
        Router request instance.
    """
    return RouterRequest(
        request_id="req-tier",
        session_id="session-tier",
        messages=[ChatMessage(content="hello")],
        max_tokens=max_tokens,
    )


def _signals(
    complexity_score: float,
    domain_tag: DomainTag = DomainTag.GENERAL,
    prompt_tokens_estimate: int = 100,
) -> TaskSignals:
    """Build task signals for complexity-tier tests.

    Args:
        complexity_score: Complexity score in ``[0, 1]``.
        domain_tag: Domain tag to route for.
        prompt_tokens_estimate: Estimated prompt tokens.

    Returns:
        Task signals instance.
    """
    return TaskSignals(
        complexity_score=complexity_score,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=prompt_tokens_estimate,
    )


def test_complexity_tier_routes_trivial_prompt_to_cheapest_model() -> None:
    """A near-zero complexity admits every model and picks the cheapest."""
    strategy = ComplexityTierStrategy(default_model_catalog())

    decision = strategy.choose(_request(), _signals(0.0))

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert decision.routing_strategy == RoutingStrategyName.COMPLEXITY_TIER
    assert "complexity-tier admitted" in decision.rationale


def test_complexity_tier_escalates_quality_with_complexity() -> None:
    """A harder prompt must not route to a lower-quality model than a trivial one."""
    catalog = default_model_catalog()
    strategy = ComplexityTierStrategy(catalog)

    trivial = strategy.choose(_request(), _signals(0.1))
    hard = strategy.choose(_request(), _signals(0.9))

    assert catalog[hard.chosen_model].quality_score >= catalog[trivial.chosen_model].quality_score
    # A 0.9 target must admit only models at or above that quality.
    assert catalog[hard.chosen_model].quality_score >= 0.9


def test_complexity_tier_admits_cheapest_model_meeting_target() -> None:
    """Among models meeting the target, the cheapest estimated cost wins."""
    catalog = default_model_catalog()
    strategy = ComplexityTierStrategy(catalog)

    decision = strategy.choose(_request(), _signals(0.9))
    chosen = catalog[decision.chosen_model]

    cheapest_qualifying = min(
        (
            candidate
            for candidate in catalog.values()
            if DomainTag.GENERAL in candidate.supports_domains and candidate.quality_score >= 0.9
        ),
        key=lambda candidate: candidate.estimate_cost(100, 256),
    )
    assert decision.chosen_model == cheapest_qualifying.model
    assert chosen.quality_score >= 0.9


def test_complexity_tier_falls_back_to_top_quality_when_target_unreachable() -> None:
    """An impossible target (>max quality) falls back to the best eligible model."""
    catalog = default_model_catalog()
    strategy = ComplexityTierStrategy(catalog)

    decision = strategy.choose(_request(), _signals(1.0))

    best_quality = max(
        candidate.quality_score
        for candidate in catalog.values()
        if DomainTag.GENERAL in candidate.supports_domains
    )
    assert catalog[decision.chosen_model].quality_score == best_quality
    assert "found no model" in decision.rationale


def test_complexity_tier_respects_domain_support() -> None:
    """Only domain-capable candidates may be selected for a domain-tagged task."""
    catalog = default_model_catalog()
    strategy = ComplexityTierStrategy(catalog)

    decision = strategy.choose(_request(), _signals(0.5, DomainTag.MEDICAL))

    assert DomainTag.MEDICAL in catalog[decision.chosen_model].supports_domains


def test_complexity_tier_is_deterministic() -> None:
    """Identical inputs must yield the same routing decision."""
    strategy = ComplexityTierStrategy(default_model_catalog())

    first = strategy.choose(_request(), _signals(0.5))
    second = strategy.choose(_request(), _signals(0.5))

    assert first.chosen_model == second.chosen_model
