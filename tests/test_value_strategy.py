"""Tests for the value (quality-per-dollar) routing strategy."""

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
from router.strategies import ValueStrategy


def _request(max_tokens: int = 256) -> RouterRequest:
    """Build a minimal router request for value-strategy tests.

    Args:
        max_tokens: Maximum output tokens for the request.

    Returns:
        Router request instance.
    """
    return RouterRequest(
        request_id="req-value",
        session_id="session-value",
        messages=[ChatMessage(content="hello")],
        max_tokens=max_tokens,
    )


def _signals(
    domain_tag: DomainTag = DomainTag.GENERAL, prompt_tokens_estimate: int = 100
) -> TaskSignals:
    """Build task signals for value-strategy tests.

    Args:
        domain_tag: Domain tag to route for.
        prompt_tokens_estimate: Estimated prompt tokens.

    Returns:
        Task signals instance.
    """
    return TaskSignals(
        complexity_score=0.4,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=prompt_tokens_estimate,
    )


def test_value_prefers_high_quality_per_dollar_model() -> None:
    """A general prompt should route to the strong-but-cheap balanced model.

    The balanced ``gpt-4.1-mini`` (quality 0.84 at a fraction of the frontier
    price) has a far higher quality-per-dollar ratio than premium models whose
    marginal quality does not justify their marginal cost.
    """
    strategy = ValueStrategy(default_model_catalog())

    decision = strategy.choose(_request(), _signals(DomainTag.GENERAL))

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert decision.routing_strategy == RoutingStrategyName.VALUE
    assert "quality-per-dollar" in decision.rationale


def test_value_respects_domain_support() -> None:
    """Only domain-capable candidates may be selected for a domain-tagged task."""
    catalog = default_model_catalog()
    strategy = ValueStrategy(catalog)

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert DomainTag.MEDICAL in catalog[decision.chosen_model].supports_domains


def test_value_is_deterministic() -> None:
    """Identical inputs must yield the same routing decision."""
    strategy = ValueStrategy(default_model_catalog())

    first = strategy.choose(_request(), _signals(DomainTag.GENERAL))
    second = strategy.choose(_request(), _signals(DomainTag.GENERAL))

    assert first.chosen_model == second.chosen_model
