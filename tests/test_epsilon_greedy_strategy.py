"""Tests for the epsilon-greedy explore/exploit routing strategy."""

from hashlib import sha256

import pytest

from router.config import default_model_catalog
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import EpsilonGreedyStrategy


def _request(request_id: str = "req-epsilon") -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id=request_id,
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


def _bucket(request_id: str) -> float:
    """Reproduce the strategy's explore/exploit bucket for a request id."""
    digest = sha256(request_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def test_epsilon_greedy_exploits_highest_quality_when_epsilon_zero() -> None:
    """With epsilon=0 every request must exploit the top-quality eligible model."""
    catalog = default_model_catalog()
    strategy = EpsilonGreedyStrategy(catalog, epsilon=0.0)
    signals = _signals()

    decision = strategy.choose(_request(), signals)

    eligible = [
        candidate
        for candidate in catalog.values()
        if DomainTag.GENERAL in candidate.supports_domains
    ]
    best = max(eligible, key=lambda candidate: (candidate.quality_score, candidate.model))
    assert decision.chosen_model == best.model
    assert decision.routing_strategy is RoutingStrategyName.EPSILON_GREEDY
    assert "exploit" in decision.rationale


def test_epsilon_greedy_explores_uniformly_when_epsilon_one() -> None:
    """With epsilon=1.0 every request explores a domain-eligible arm."""
    catalog = default_model_catalog()
    strategy = EpsilonGreedyStrategy(catalog, epsilon=1.0)
    signals = _signals()

    decision = strategy.choose(_request("req-explore-arm"), signals)

    eligible_names = {
        candidate.model
        for candidate in catalog.values()
        if DomainTag.GENERAL in candidate.supports_domains
    }
    assert decision.chosen_model in eligible_names
    assert "explore" in decision.rationale


def test_epsilon_greedy_respects_domain_eligibility() -> None:
    """Only domain-eligible models may be chosen for a specialized domain."""
    catalog = default_model_catalog()
    strategy = EpsilonGreedyStrategy(catalog, epsilon=0.0)

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert DomainTag.MEDICAL in catalog[decision.chosen_model].supports_domains


def test_epsilon_greedy_is_deterministic() -> None:
    """Identical inputs must yield an identical decision (replay-safe)."""
    catalog = default_model_catalog()
    strategy = EpsilonGreedyStrategy(catalog, epsilon=0.1)

    first = strategy.choose(_request("req-stable"), _signals())
    second = strategy.choose(_request("req-stable"), _signals())

    assert first.chosen_model == second.chosen_model
    assert first.rationale == second.rationale
    assert first.fallback_chain == second.fallback_chain


def test_epsilon_greedy_bucket_boundary_matches_hash() -> None:
    """The explore/exploit split must follow the canary-style request_id hash."""
    catalog = default_model_catalog()
    epsilon = 0.25
    strategy = EpsilonGreedyStrategy(catalog, epsilon=epsilon)

    # Find one request that explores and one that exploits under this epsilon.
    explore_id = next(f"req-eg-{i}" for i in range(10_000) if _bucket(f"req-eg-{i}") < epsilon)
    exploit_id = next(f"req-eg-{i}" for i in range(10_000) if _bucket(f"req-eg-{i}") >= epsilon)

    explore = strategy.choose(_request(explore_id), _signals())
    exploit = strategy.choose(_request(exploit_id), _signals())

    assert "explore" in explore.rationale
    assert "exploit" in exploit.rationale


def test_epsilon_greedy_rejects_out_of_range_epsilon() -> None:
    """Epsilon outside [0.0, 1.0] should fail fast at construction."""
    with pytest.raises(ValueError, match="epsilon must be within"):
        EpsilonGreedyStrategy(default_model_catalog(), epsilon=1.5)
