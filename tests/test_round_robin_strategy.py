"""Tests for the round-robin provider load-balancing strategy."""

from collections import Counter

from router.config import default_model_catalog
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import RoundRobinStrategy


def _request(request_id: str) -> RouterRequest:
    """Build a minimal router request with a given request id."""
    return RouterRequest(request_id=request_id, messages=[ChatMessage(content="Hello")])


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build general-purpose task signals for a domain."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_round_robin_is_deterministic_for_same_request_id() -> None:
    """The same request id must always resolve to the same model (replay-safe)."""
    strategy = RoundRobinStrategy(default_model_catalog())
    first = strategy.choose(_request("req-42"), _signals())
    second = strategy.choose(_request("req-42"), _signals())

    assert first.chosen_model == second.chosen_model
    assert first.routing_strategy is RoutingStrategyName.ROUND_ROBIN


def test_round_robin_spreads_across_all_eligible_providers() -> None:
    """Distinct request ids must spread across every domain-eligible provider."""
    catalog = default_model_catalog()
    strategy = RoundRobinStrategy(catalog)
    eligible_providers = {
        candidate.provider
        for candidate in catalog.values()
        if DomainTag.GENERAL in candidate.supports_domains
    }

    chosen_providers = Counter(
        strategy.choose(_request(f"req-{index}"), _signals()).provider for index in range(400)
    )

    assert set(chosen_providers) == eligible_providers
    # With 400 ids hashed across the providers, every provider should get a
    # non-trivial slice rather than the pool collapsing onto one provider.
    assert all(count > 0 for count in chosen_providers.values())


def test_round_robin_picks_best_quality_model_within_provider() -> None:
    """Within the balanced provider, the highest-quality eligible model wins."""
    catalog = default_model_catalog()
    strategy = RoundRobinStrategy(catalog)

    for index in range(50):
        decision = strategy.choose(_request(f"req-{index}"), _signals())
        provider = catalog[decision.chosen_model].provider
        best_for_provider = max(
            (
                candidate
                for candidate in catalog.values()
                if candidate.provider == provider
                and DomainTag.GENERAL in candidate.supports_domains
            ),
            key=lambda candidate: candidate.quality_score,
        )
        assert decision.chosen_model == best_for_provider.model


def test_round_robin_respects_domain_eligibility() -> None:
    """Only providers with a domain-eligible model may be selected."""
    catalog = default_model_catalog()
    strategy = RoundRobinStrategy(catalog)
    medical_providers = {
        candidate.provider
        for candidate in catalog.values()
        if DomainTag.MEDICAL in candidate.supports_domains
    }

    for index in range(100):
        decision = strategy.choose(_request(f"med-{index}"), _signals(DomainTag.MEDICAL))
        assert catalog[decision.chosen_model].provider in medical_providers
        assert DomainTag.MEDICAL in catalog[decision.chosen_model].supports_domains
