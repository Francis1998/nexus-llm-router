"""Tests for the sticky-session routing strategy."""

from router.config import default_model_catalog
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import StickySessionStrategy


def _request(session_id: str = "session-1", request_id: str = "req-1") -> RouterRequest:
    """Build a minimal router request for strategy tests.

    Args:
        session_id: Session identifier used for sticky pinning.
        request_id: Unique request identifier.

    Returns:
        Router request instance.
    """
    return RouterRequest(
        request_id=request_id,
        session_id=session_id,
        messages=[ChatMessage(content="hello")],
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests.

    Args:
        domain_tag: Domain tag to route for.

    Returns:
        Task signals instance.
    """
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_sticky_session_is_deterministic_within_a_session() -> None:
    """Requests sharing a session_id must route to the same model."""
    strategy = StickySessionStrategy(default_model_catalog())

    first = strategy.choose(_request(session_id="alpha", request_id="req-1"), _signals())
    second = strategy.choose(_request(session_id="alpha", request_id="req-2"), _signals())

    assert first.chosen_model == second.chosen_model
    assert first.routing_strategy == RoutingStrategyName.STICKY_SESSION
    assert "alpha" in first.rationale


def test_sticky_session_pins_regardless_of_request_id() -> None:
    """The pin depends only on session_id, not the per-request id."""
    strategy = StickySessionStrategy(default_model_catalog())

    decisions = {
        strategy.choose(
            _request(session_id="beta", request_id=f"req-{index}"), _signals()
        ).chosen_model
        for index in range(5)
    }

    assert len(decisions) == 1


def test_sticky_session_distributes_distinct_sessions() -> None:
    """Distinct sessions should spread across more than one model."""
    strategy = StickySessionStrategy(default_model_catalog())

    chosen_models = {
        strategy.choose(_request(session_id=f"session-{index}"), _signals()).chosen_model
        for index in range(50)
    }

    assert len(chosen_models) > 1


def test_sticky_session_respects_domain_support() -> None:
    """Only domain-capable models may be pinned for a domain-tagged session."""
    catalog = default_model_catalog()
    strategy = StickySessionStrategy(catalog)

    for index in range(50):
        decision = strategy.choose(_request(session_id=f"med-{index}"), _signals(DomainTag.MEDICAL))
        assert DomainTag.MEDICAL in catalog[decision.chosen_model].supports_domains
