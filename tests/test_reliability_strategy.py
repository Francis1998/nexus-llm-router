"""Tests for the reliability-aware routing strategy."""

from router.config import default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_FRONTIER_MODEL
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import ReliabilityAwareStrategy
from safety.circuit_breaker import CircuitBreakerRegistry


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


def test_reliability_prefers_highest_quality_healthy_provider() -> None:
    """The top-quality model is chosen when its provider is healthy."""
    strategy = ReliabilityAwareStrategy(default_model_catalog(), _FakeHealth(set()))

    decision = strategy.choose(_request(), _signals())

    # anthropic (0.98) is highest-quality general provider and is healthy.
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy == RoutingStrategyName.RELIABILITY_AWARE
    assert "healthy provider" in decision.rationale


def test_reliability_routes_around_unhealthy_provider() -> None:
    """A model on an open circuit is skipped for a healthy alternative."""
    strategy = ReliabilityAwareStrategy(
        default_model_catalog(), _FakeHealth({"anthropic", "google"})
    )

    decision = strategy.choose(_request(), _signals())

    # With anthropic and google down, openai frontier (0.97) is the best healthy pick.
    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"


def test_reliability_fallback_chain_orders_healthy_first() -> None:
    """Fallback candidates on healthy providers must precede unhealthy ones."""
    catalog = default_model_catalog()
    strategy = ReliabilityAwareStrategy(catalog, _FakeHealth({"anthropic"}))

    decision = strategy.choose(_request(), _signals())

    providers = [catalog[model].provider for model in decision.fallback_chain]
    healthy_flags = [provider != "anthropic" for provider in providers]
    # Healthy providers (True) must come before any unhealthy provider (False).
    assert healthy_flags == sorted(healthy_flags, reverse=True)


def test_reliability_uses_highest_quality_when_all_unhealthy() -> None:
    """When no provider is healthy, the highest-quality model is still returned."""
    all_providers = {"openai", "anthropic", "google", "moonshot"}
    strategy = ReliabilityAwareStrategy(default_model_catalog(), _FakeHealth(all_providers))

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "no healthy provider" in decision.rationale


def test_reliability_strategy_reads_live_circuit_breaker_state() -> None:
    """The strategy honors a real circuit breaker once it trips open."""
    registry = CircuitBreakerRegistry(failure_threshold=2, recovery_window_seconds=60.0)
    strategy = ReliabilityAwareStrategy(default_model_catalog(), registry)

    for _ in range(2):
        registry.record_failure("anthropic")

    decision = strategy.choose(_request(), _signals())

    assert decision.provider != "anthropic"
