"""Tests for the adaptive-exploration decaying epsilon routing strategy."""

from hashlib import sha256

import pytest

from router.config import RouterSettings, default_model_catalog
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    AdaptiveExplorationStrategy,
    InflightStats,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(request_id: str = "req-adaptive-exploration") -> RouterRequest:
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


def test_adaptive_exploration_cold_start_uses_base_epsilon() -> None:
    """With no successes the effective epsilon equals the configured base."""
    strategy = AdaptiveExplorationStrategy(
        default_model_catalog(),
        SuccessStats(),
        base_epsilon=0.2,
        min_epsilon=0.02,
    )

    assert strategy.current_epsilon() == pytest.approx(0.2)


def test_adaptive_exploration_decays_toward_min_as_successes_grow() -> None:
    """More recorded successes must shrink epsilon toward the floor."""
    stats = SuccessStats()
    strategy = AdaptiveExplorationStrategy(
        default_model_catalog(),
        stats,
        base_epsilon=0.2,
        min_epsilon=0.02,
    )
    for _ in range(9):
        stats.observe("openai", success=True)

    # epsilon = 0.02 + (0.2 - 0.02) / (1 + 9) = 0.038
    assert strategy.current_epsilon() == pytest.approx(0.038)
    assert SuccessStats().total_successes() == 0
    assert stats.total_successes() == 9


def test_adaptive_exploration_exploits_highest_quality_when_epsilon_zero() -> None:
    """With base=min=0 every request must exploit the top-quality eligible model."""
    catalog = default_model_catalog()
    strategy = AdaptiveExplorationStrategy(
        catalog,
        SuccessStats(),
        base_epsilon=0.0,
        min_epsilon=0.0,
    )
    signals = _signals()

    decision = strategy.choose(_request(), signals)

    eligible = [
        candidate
        for candidate in catalog.values()
        if DomainTag.GENERAL in candidate.supports_domains
    ]
    best = max(eligible, key=lambda candidate: (candidate.quality_score, candidate.model))
    assert decision.chosen_model == best.model
    assert decision.routing_strategy is RoutingStrategyName.ADAPTIVE_EXPLORATION
    assert "exploit" in decision.rationale


def test_adaptive_exploration_bucket_boundary_matches_decayed_epsilon() -> None:
    """Explore/exploit split must follow the canary-style hash against live epsilon."""
    catalog = default_model_catalog()
    stats = SuccessStats()
    for _ in range(9):
        stats.observe("anthropic", success=True)
    strategy = AdaptiveExplorationStrategy(
        catalog,
        stats,
        base_epsilon=0.2,
        min_epsilon=0.02,
    )
    epsilon = strategy.current_epsilon()

    explore_id = next(f"req-ae-{i}" for i in range(10_000) if _bucket(f"req-ae-{i}") < epsilon)
    exploit_id = next(f"req-ae-{i}" for i in range(10_000) if _bucket(f"req-ae-{i}") >= epsilon)

    explore = strategy.choose(_request(explore_id), _signals())
    exploit = strategy.choose(_request(exploit_id), _signals())

    assert "explore" in explore.rationale
    assert "exploit" in exploit.rationale
    assert "successes=9" in explore.rationale


def test_adaptive_exploration_respects_domain_eligibility() -> None:
    """Only domain-eligible models may be chosen for a specialized domain."""
    strategy = AdaptiveExplorationStrategy(
        default_model_catalog(),
        SuccessStats(),
        base_epsilon=0.0,
        min_epsilon=0.0,
    )

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert DomainTag.MEDICAL in default_model_catalog()[decision.chosen_model].supports_domains


def test_adaptive_exploration_is_deterministic() -> None:
    """Identical inputs must yield an identical decision (replay-safe)."""
    strategy = AdaptiveExplorationStrategy(
        default_model_catalog(),
        SuccessStats(),
        base_epsilon=0.2,
        min_epsilon=0.02,
    )

    first = strategy.choose(_request("req-stable-ae"), _signals())
    second = strategy.choose(_request("req-stable-ae"), _signals())

    assert first.chosen_model == second.chosen_model
    assert first.rationale == second.rationale
    assert first.fallback_chain == second.fallback_chain


def test_adaptive_exploration_rejects_invalid_epsilons() -> None:
    """Out-of-range or inverted base/min must fail fast at construction."""
    catalog = default_model_catalog()
    stats = SuccessStats()
    with pytest.raises(ValueError, match="base_epsilon must be within"):
        AdaptiveExplorationStrategy(catalog, stats, base_epsilon=1.5, min_epsilon=0.02)
    with pytest.raises(ValueError, match="min_epsilon must be within"):
        AdaptiveExplorationStrategy(catalog, stats, base_epsilon=0.2, min_epsilon=-0.1)
    with pytest.raises(ValueError, match="must be <="):
        AdaptiveExplorationStrategy(catalog, stats, base_epsilon=0.02, min_epsilon=0.2)


def test_adaptive_exploration_enum_parses() -> None:
    """The strategy name string must resolve to the enum member."""
    assert RoutingStrategyName("adaptive-exploration") is RoutingStrategyName.ADAPTIVE_EXPLORATION


def test_adaptive_exploration_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose adaptive-exploration with SuccessStats."""
    settings = RouterSettings()
    strategies = build_strategies(
        default_model_catalog(),
        LatencyStats(),
        InflightStats(),
        settings.quality_floor,
        settings.ab_model_a,
        settings.ab_model_b,
        settings.ab_model_a_weight,
        CircuitBreakerRegistry(),
        settings.blend_quality_weight,
        settings.blend_cost_weight,
        settings.blend_latency_weight,
        settings.request_cost_ceiling_usd,
        settings.canary_stable_model,
        settings.canary_model,
        settings.canary_weight,
        settings.latency_sla_ms,
        settings.prompt_prefix_cache_min_chars,
        settings.epsilon,
        settings.availability_slo,
        SuccessStats(),
        settings.failover_priority,
        settings.health_blend_success_weight,
        settings.health_blend_latency_weight,
        settings.health_blend_quality_weight,
        settings.health_blend_cost_weight,
        adaptive_exploration_base=settings.adaptive_exploration_base,
        adaptive_exploration_min=settings.adaptive_exploration_min,
    )

    strategy = strategies[RoutingStrategyName.ADAPTIVE_EXPLORATION]
    assert isinstance(strategy, AdaptiveExplorationStrategy)
    assert strategy.strategy_name is RoutingStrategyName.ADAPTIVE_EXPLORATION
    assert strategy.current_epsilon() == pytest.approx(0.2)
