"""Tests for the health-cost-latency routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_PRO_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
)
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    ModelCandidate,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    HealthCostLatencyStrategy,
    InflightStats,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(max_tokens: int = 512) -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id="req-hcl",
        messages=[ChatMessage(content="Summarize the incident and next steps.")],
        max_tokens=max_tokens,
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for a domain."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _strategy(
    *,
    success_stats: SuccessStats | None = None,
    latency_stats: LatencyStats | None = None,
    health_weight: float = 0.4,
    cost_weight: float = 0.3,
    latency_weight: float = 0.3,
    catalog: dict[str, ModelCandidate] | None = None,
) -> HealthCostLatencyStrategy:
    """Build a health/cost/latency blend strategy with overridable dependencies."""
    return HealthCostLatencyStrategy(
        catalog or default_model_catalog(),
        success_stats or SuccessStats(),
        latency_stats or LatencyStats(),
        health_weight=health_weight,
        cost_weight=cost_weight,
        latency_weight=latency_weight,
    )


def test_hcl_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("health-cost-latency") is RoutingStrategyName.HEALTH_COST_LATENCY


def test_hcl_health_weight_prefers_reliable_provider() -> None:
    """A degraded provider loses when success rate dominates."""
    stats = SuccessStats()
    stats.observe("anthropic", success=False)
    stats.observe("openai", success=True)
    stats.observe("openai", success=True)
    strategy = _strategy(
        success_stats=stats,
        health_weight=1.0,
        cost_weight=0.0,
        latency_weight=0.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.provider == "openai"
    assert decision.chosen_model in {OPENAI_FRONTIER_MODEL, OPENAI_BALANCED_MODEL}
    assert decision.routing_strategy is RoutingStrategyName.HEALTH_COST_LATENCY
    assert "health=100.00%" in decision.rationale


def test_hcl_cost_weight_prefers_cheaper_model() -> None:
    """With all weight on cost, the cheapest general model is chosen."""
    strategy = _strategy(
        health_weight=0.0,
        cost_weight=1.0,
        latency_weight=0.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL


def test_hcl_latency_weight_prefers_faster_provider() -> None:
    """Latency weighting steers selection toward the lower-p95 provider."""
    latency_stats = LatencyStats()
    for _ in range(5):
        latency_stats.observe("anthropic", 4000.0)
        latency_stats.observe("openai", 4000.0)
        latency_stats.observe("google", 4000.0)
        latency_stats.observe("moonshot", 10.0)
    strategy = _strategy(
        latency_stats=latency_stats,
        health_weight=0.0,
        cost_weight=0.0,
        latency_weight=1.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_hcl_empty_latency_and_equal_costs_do_not_divide_by_zero() -> None:
    """Equal cost/latency normalization should be neutral and safe."""
    catalog = {
        "steady-a": ModelCandidate(
            model="steady-a",
            provider="steady-a",
            quality_score=0.70,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.001,
            supports_domains={DomainTag.GENERAL},
        ),
        "steady-b": ModelCandidate(
            model="steady-b",
            provider="steady-b",
            quality_score=0.90,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.001,
            supports_domains={DomainTag.GENERAL},
        ),
    }
    stats = SuccessStats()
    stats.observe("steady-a", success=True)
    stats.observe("steady-b", success=False)
    strategy = _strategy(
        catalog=catalog,
        success_stats=stats,
        health_weight=1.0,
        cost_weight=0.5,
        latency_weight=0.5,
    )

    decision = strategy.choose(_request(max_tokens=128), _signals())

    assert decision.chosen_model == "steady-a"
    assert "p95=0.0ms" in decision.rationale


def test_hcl_zero_weights_fall_back_to_health() -> None:
    """All-zero weights degrade to pure health instead of failing."""
    stats = SuccessStats()
    stats.observe("anthropic", success=False)
    stats.observe("openai", success=True)
    strategy = _strategy(
        success_stats=stats,
        health_weight=0.0,
        cost_weight=0.0,
        latency_weight=0.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.provider == "openai"
    assert decision.chosen_model in {OPENAI_FRONTIER_MODEL, OPENAI_BALANCED_MODEL}


def test_hcl_rejects_negative_weight() -> None:
    """Negative weights should fail fast at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        _strategy(health_weight=-0.1)


def test_hcl_respects_domain_eligibility() -> None:
    """Unsupported domains must not win even when they score better on other axes."""
    strategy = _strategy()

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert DomainTag.MEDICAL in default_model_catalog()[decision.chosen_model].supports_domains
    assert decision.chosen_model in {ANTHROPIC_SAFETY_MODEL, GEMINI_PRO_MODEL}


def test_hcl_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose health-cost-latency."""
    catalog = default_model_catalog()
    settings = RouterSettings(
        hcl_health_weight=0.5,
        hcl_cost_weight=0.25,
        hcl_latency_weight=0.25,
    )
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
        hcl_health_weight=settings.hcl_health_weight,
        hcl_cost_weight=settings.hcl_cost_weight,
        hcl_latency_weight=settings.hcl_latency_weight,
    )

    assert (
        strategies[RoutingStrategyName.HEALTH_COST_LATENCY].strategy_name
        is RoutingStrategyName.HEALTH_COST_LATENCY
    )
