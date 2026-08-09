"""Tests for the queue-depth-fairness routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_FRONTIER_MODEL
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
    QueueDepthFairnessStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id="req-queue-depth-fair",
        messages=[ChatMessage(content="Balance queue depth fairly.")],
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _strategy(
    *,
    inflight_stats: InflightStats | None = None,
    queue_depth_soft_cap: int = 4,
) -> QueueDepthFairnessStrategy:
    """Build queue-depth-fairness with overridable dependencies."""
    return QueueDepthFairnessStrategy(
        default_model_catalog(),
        inflight_stats or InflightStats(),
        queue_depth_soft_cap=queue_depth_soft_cap,
    )


def test_queue_depth_fairness_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("queue-depth-fairness") is RoutingStrategyName.QUEUE_DEPTH_FAIRNESS


def test_queue_depth_fairness_cold_start_picks_top_quality() -> None:
    """Cold InflightStats treat every provider as depth 0 under the soft cap."""
    strategy = _strategy()

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.QUEUE_DEPTH_FAIRNESS
    assert "under soft cap 4" in decision.rationale
    assert "depth 0/4" in decision.rationale


def test_queue_depth_fairness_sheds_deep_top_provider() -> None:
    """A deep top-quality provider loses to a shallower alternative under the cap."""
    stats = InflightStats()
    for _ in range(4):
        stats.begin("anthropic")
    strategy = _strategy(inflight_stats=stats, queue_depth_soft_cap=4)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "under soft cap 4" in decision.rationale
    assert ANTHROPIC_SAFETY_MODEL not in decision.fallback_chain[:1]


def test_queue_depth_fairness_all_over_cap_uses_lowest_depth() -> None:
    """When every provider is at/over the soft cap, lowest depth wins."""
    stats = InflightStats()
    for _ in range(6):
        stats.begin("anthropic")
    for _ in range(5):
        stats.begin("openai")
    for _ in range(4):
        stats.begin("google")
    for _ in range(7):
        stats.begin("moonshot")
    strategy = _strategy(inflight_stats=stats, queue_depth_soft_cap=4)

    decision = strategy.choose(_request(), _signals())

    assert decision.provider == "google"
    assert "every eligible provider at or above soft cap 4" in decision.rationale
    assert "lowest-depth fallback google depth 4/4" in decision.rationale


def test_queue_depth_fairness_rejects_invalid_soft_cap() -> None:
    """Invalid soft caps fail fast at construction."""
    with pytest.raises(ValueError, match="queue_depth_soft_cap must be >= 1"):
        _strategy(queue_depth_soft_cap=0)


def test_queue_depth_fairness_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose queue-depth-fairness."""
    settings = RouterSettings(queue_depth_soft_cap=3)
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
        settings.concurrency_cap,
        provider_family_cost_ceiling_usd=settings.provider_family_cost_ceiling_usd,
        cache_hit_sticky_min_chars=settings.cache_hit_sticky_min_chars,
        tenant_concurrency_lease=settings.tenant_concurrency_lease,
        provider_error_budget_rate=settings.provider_error_budget_rate,
        queue_depth_soft_cap=settings.queue_depth_soft_cap,
    )

    strategy = strategies[RoutingStrategyName.QUEUE_DEPTH_FAIRNESS]
    assert isinstance(strategy, QueueDepthFairnessStrategy)
    assert strategy.strategy_name is RoutingStrategyName.QUEUE_DEPTH_FAIRNESS


def test_queue_depth_fairness_settings_default() -> None:
    """RouterSettings expose the queue-depth soft-cap default."""
    assert RouterSettings().queue_depth_soft_cap == 4
