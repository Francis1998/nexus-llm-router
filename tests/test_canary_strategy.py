"""Tests for the health-gated canary routing strategy."""

import pytest

from router.config import default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_BALANCED_MODEL
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import CanaryStrategy


class _FakeHealth:
    """Deterministic provider-health view for unit tests."""

    def __init__(self, unavailable: set[str]) -> None:
        """Store the set of providers considered unavailable.

        Args:
            unavailable: Provider names whose circuits are open.
        """
        self._unavailable = unavailable

    def is_available(self, provider: str) -> bool:
        """Return whether a provider is routable.

        Args:
            provider: Provider name.

        Returns:
            True unless the provider was marked unavailable.
        """
        return provider not in self._unavailable


def _request(request_id: str = "req-1") -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(request_id=request_id, messages=[ChatMessage(content="hello")])


def _signals() -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def _strategy(canary_weight: float, unavailable: set[str] | None = None) -> CanaryStrategy:
    """Build a canary strategy with distinct stable/canary providers."""
    return CanaryStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable or set()),
        stable_model=OPENAI_BALANCED_MODEL,
        canary_model=ANTHROPIC_SAFETY_MODEL,
        canary_weight=canary_weight,
    )


def test_canary_routes_all_traffic_to_canary_at_full_weight() -> None:
    """A weight of 1.0 sends every healthy request to the canary model."""
    strategy = _strategy(canary_weight=1.0)

    decision = strategy.choose(_request("req-abc"), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy == RoutingStrategyName.CANARY
    assert "routed to canary" in decision.rationale


def test_canary_routes_all_traffic_to_stable_at_zero_weight() -> None:
    """A weight of 0.0 keeps every request on the stable model."""
    strategy = _strategy(canary_weight=0.0)

    decision = strategy.choose(_request("req-abc"), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "routed to stable" in decision.rationale


def test_canary_pauses_when_canary_provider_unhealthy() -> None:
    """Even at full weight, an unhealthy canary provider routes to stable.

    This is the key difference from the symmetric A/B strategy: a failing
    canary must not keep drawing its share of live traffic.
    """
    strategy = _strategy(canary_weight=1.0, unavailable={"anthropic"})

    decision = strategy.choose(_request("req-abc"), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "canary paused" in decision.rationale


def test_canary_bucketing_is_deterministic_for_a_request_id() -> None:
    """The same request id always resolves to the same arm."""
    strategy = _strategy(canary_weight=0.5)

    first = strategy.choose(_request("stable-session"), _signals())
    second = strategy.choose(_request("stable-session"), _signals())

    assert first.chosen_model == second.chosen_model


def test_canary_fallback_chain_anchors_on_stable_model() -> None:
    """When the canary is chosen, the stable model is the first fallback."""
    strategy = _strategy(canary_weight=1.0)

    decision = strategy.choose(_request("req-abc"), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.fallback_chain[0] == OPENAI_BALANCED_MODEL


def test_canary_rejects_unknown_models() -> None:
    """Canary arms outside the catalog fail fast at construction."""
    with pytest.raises(ValueError, match="not in model catalog"):
        CanaryStrategy(
            default_model_catalog(),
            _FakeHealth(set()),
            stable_model="nonexistent-model",
            canary_model=ANTHROPIC_SAFETY_MODEL,
            canary_weight=0.1,
        )


def test_canary_rejects_out_of_range_weight() -> None:
    """Canary weights outside [0.0, 1.0] fail fast at construction."""
    with pytest.raises(ValueError, match="canary_weight must be within"):
        CanaryStrategy(
            default_model_catalog(),
            _FakeHealth(set()),
            stable_model=OPENAI_BALANCED_MODEL,
            canary_model=ANTHROPIC_SAFETY_MODEL,
            canary_weight=1.5,
        )
