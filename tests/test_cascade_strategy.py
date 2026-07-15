"""Tests for the cascade (cheapest-first escalation) routing strategy."""

from router.config import default_model_catalog
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import CascadeStrategy


def _request(request_id: str = "req-cascade", max_tokens: int = 256) -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id=request_id,
        messages=[ChatMessage(content="Hello")],
        max_tokens=max_tokens,
    )


def _signals(
    domain_tag: DomainTag = DomainTag.GENERAL,
    latency_requirement: LatencyRequirement = LatencyRequirement.REALTIME,
) -> TaskSignals:
    """Build task signals for a domain and latency requirement."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=latency_requirement,
        token_budget=4096,
        prompt_tokens_estimate=64,
    )


def test_cascade_routes_to_cheapest_eligible_model() -> None:
    """The primary attempt must be the cheapest domain-eligible model."""
    catalog = default_model_catalog()
    strategy = CascadeStrategy(catalog)
    request, signals = _request(), _signals()

    decision = strategy.choose(request, signals)

    eligible = [
        candidate
        for candidate in catalog.values()
        if DomainTag.GENERAL in candidate.supports_domains and candidate.supports_realtime
    ]
    cheapest = min(
        eligible,
        key=lambda candidate: candidate.estimate_cost(
            signals.prompt_tokens_estimate, request.max_tokens
        ),
    )
    assert decision.chosen_model == cheapest.model
    assert decision.routing_strategy is RoutingStrategyName.CASCADE


def test_cascade_fallback_chain_ascends_in_cost() -> None:
    """The fallback ladder must be ordered by non-decreasing estimated cost.

    This is the strategy's defining behaviour: unlike the base quality-ordered
    fallback, a cascade escalates one price/capability rung at a time, so each
    fallback entry must cost at least as much as the primary and the ones before
    it.
    """
    catalog = default_model_catalog()
    strategy = CascadeStrategy(catalog)
    request, signals = _request(), _signals()

    decision = strategy.choose(request, signals)
    ladder = [decision.chosen_model, *decision.fallback_chain]
    costs = [
        catalog[model].estimate_cost(signals.prompt_tokens_estimate, request.max_tokens)
        for model in ladder
    ]

    assert costs == sorted(costs)
    assert decision.chosen_model not in decision.fallback_chain


def test_cascade_respects_domain_eligibility() -> None:
    """Only domain-eligible models may be chosen for a specialized domain."""
    catalog = default_model_catalog()
    strategy = CascadeStrategy(catalog)

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert DomainTag.MEDICAL in catalog[decision.chosen_model].supports_domains


def test_cascade_is_deterministic() -> None:
    """Identical inputs must yield an identical decision (replay-safe)."""
    catalog = default_model_catalog()
    strategy = CascadeStrategy(catalog)

    first = strategy.choose(_request(), _signals())
    second = strategy.choose(_request(), _signals())

    assert first.chosen_model == second.chosen_model
    assert first.fallback_chain == second.fallback_chain
