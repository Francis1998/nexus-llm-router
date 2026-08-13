"""Tests for the quality-weighted-sticky routing strategy."""

from collections import Counter

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    InflightStats,
    LatencyStats,
    QualityWeightedStickyStrategy,
    StickySessionStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(session_id: str = "session-1", request_id: str = "req-1") -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(
        request_id=request_id,
        session_id=session_id,
        messages=[ChatMessage(content="hello")],
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_quality_weighted_sticky_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("quality-weighted-sticky")
        is RoutingStrategyName.QUALITY_WEIGHTED_STICKY
    )


def test_quality_weighted_sticky_is_deterministic_within_a_session() -> None:
    """Requests sharing a session_id must route to the same model."""
    strategy = QualityWeightedStickyStrategy(default_model_catalog())

    first = strategy.choose(_request(session_id="alpha", request_id="req-1"), _signals())
    second = strategy.choose(_request(session_id="alpha", request_id="req-2"), _signals())

    assert first.chosen_model == second.chosen_model
    assert first.routing_strategy is RoutingStrategyName.QUALITY_WEIGHTED_STICKY
    assert "alpha" in first.rationale
    assert "quality weight" in first.rationale


def test_quality_weighted_sticky_distributes_distinct_sessions() -> None:
    """Distinct sessions should spread across more than one model."""
    strategy = QualityWeightedStickyStrategy(default_model_catalog())

    chosen_models = {
        strategy.choose(_request(session_id=f"session-{index}"), _signals()).chosen_model
        for index in range(80)
    }

    assert len(chosen_models) > 1


def test_quality_weighted_sticky_biases_toward_higher_quality() -> None:
    """Higher-quality models should win a larger share than uniform sticky.

    Claude Sonnet 4.6 leads catalog quality and should appear at least as often
    as under uniform sticky-session hashing across many session ids.
    """
    catalog = default_model_catalog()
    weighted = QualityWeightedStickyStrategy(catalog)
    uniform = StickySessionStrategy(catalog)

    weighted_counts = Counter(
        weighted.choose(_request(session_id=f"sess-{index}"), _signals()).chosen_model
        for index in range(400)
    )
    uniform_counts = Counter(
        uniform.choose(_request(session_id=f"sess-{index}"), _signals()).chosen_model
        for index in range(400)
    )

    assert weighted_counts[ANTHROPIC_SAFETY_MODEL] >= uniform_counts[ANTHROPIC_SAFETY_MODEL]


def test_quality_weighted_sticky_respects_domain_support() -> None:
    """Only domain-capable models may be pinned for a domain-tagged session."""
    catalog = default_model_catalog()
    strategy = QualityWeightedStickyStrategy(catalog)

    for index in range(50):
        decision = strategy.choose(_request(session_id=f"med-{index}"), _signals(DomainTag.MEDICAL))
        assert DomainTag.MEDICAL in catalog[decision.chosen_model].supports_domains


def test_quality_weighted_sticky_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose quality-weighted-sticky."""
    catalog = default_model_catalog()
    settings = RouterSettings()
    strategies = build_strategies(
        catalog,
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
        settings.concurrency_cap,
    )

    strategy = strategies[RoutingStrategyName.QUALITY_WEIGHTED_STICKY]
    assert isinstance(strategy, QualityWeightedStickyStrategy)
    assert strategy.strategy_name is RoutingStrategyName.QUALITY_WEIGHTED_STICKY
