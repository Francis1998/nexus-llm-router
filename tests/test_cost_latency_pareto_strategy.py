"""Tests for the cost-latency-pareto routing strategy."""

from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_PRO_MODEL,
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
    CostLatencyParetoStrategy,
    InflightStats,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(request_id="req-pareto", messages=[ChatMessage(content="hello")])


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=64,
    )


def test_cost_latency_pareto_enum_parses() -> None:
    """The strategy name should round-trip through the StrEnum."""
    assert RoutingStrategyName("cost-latency-pareto") is RoutingStrategyName.COST_LATENCY_PARETO


def test_cost_latency_pareto_excludes_dominated_candidates() -> None:
    """A strictly worse cost and latency candidate must leave the Pareto front."""
    catalog = {
        "cheap-fast": ModelCandidate(
            model="cheap-fast",
            provider="fast-provider",
            quality_score=0.7,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
        ),
        "expensive-slow": ModelCandidate(
            model="expensive-slow",
            provider="slow-provider",
            quality_score=0.99,
            input_cost_per_1k=0.050,
            output_cost_per_1k=0.100,
            supports_domains={DomainTag.GENERAL},
        ),
    }
    latency_stats = LatencyStats()
    latency_stats.observe("fast-provider", 100.0)
    latency_stats.observe("slow-provider", 900.0)
    strategy = CostLatencyParetoStrategy(catalog, latency_stats)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == "cheap-fast"
    assert decision.routing_strategy is RoutingStrategyName.COST_LATENCY_PARETO
    assert "non-dominated" in decision.rationale
    assert "front size 1" in decision.rationale


def test_cost_latency_pareto_quality_breaks_front_ties() -> None:
    """Non-dominated trade-offs resolve by higher quality on the Pareto front."""
    catalog = {
        "cheap-slow": ModelCandidate(
            model="cheap-slow",
            provider="cheap-provider",
            quality_score=0.6,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
        ),
        "pricey-fast": ModelCandidate(
            model="pricey-fast",
            provider="fast-provider",
            quality_score=0.95,
            input_cost_per_1k=0.020,
            output_cost_per_1k=0.040,
            supports_domains={DomainTag.GENERAL},
        ),
        "dominated": ModelCandidate(
            model="dominated",
            provider="dominated-provider",
            quality_score=0.99,
            input_cost_per_1k=0.030,
            output_cost_per_1k=0.060,
            supports_domains={DomainTag.GENERAL},
        ),
    }
    latency_stats = LatencyStats()
    latency_stats.observe("cheap-provider", 800.0)
    latency_stats.observe("fast-provider", 120.0)
    latency_stats.observe("dominated-provider", 500.0)
    strategy = CostLatencyParetoStrategy(catalog, latency_stats)

    decision = strategy.choose(_request(), _signals())

    # cheap-slow and pricey-fast trade off cost vs latency; dominated loses both axes.
    assert decision.chosen_model == "pricey-fast"
    assert "front size 2" in decision.rationale


def test_cost_latency_pareto_cold_start_prefers_cheapest_then_quality() -> None:
    """With equal cold latencies, only min-cost candidates stay on the front."""
    catalog = {
        "expensive-high-quality": ModelCandidate(
            model="expensive-high-quality",
            provider="expensive-provider",
            quality_score=0.99,
            input_cost_per_1k=0.010,
            output_cost_per_1k=0.020,
            supports_domains={DomainTag.GENERAL},
        ),
        "cheap-mid-quality": ModelCandidate(
            model="cheap-mid-quality",
            provider="cheap-provider",
            quality_score=0.8,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
        ),
        "cheap-low-quality": ModelCandidate(
            model="cheap-low-quality",
            provider="also-cheap-provider",
            quality_score=0.5,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
        ),
    }
    strategy = CostLatencyParetoStrategy(catalog, LatencyStats())

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == "cheap-mid-quality"
    assert "front size 2" in decision.rationale


def test_cost_latency_pareto_respects_domain_eligibility() -> None:
    """Unsupported domains must not win even when they look Pareto-optimal."""
    strategy = CostLatencyParetoStrategy(default_model_catalog(), LatencyStats())

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert DomainTag.MEDICAL in default_model_catalog()[decision.chosen_model].supports_domains
    assert decision.chosen_model in {ANTHROPIC_SAFETY_MODEL, GEMINI_PRO_MODEL}


def test_cost_latency_pareto_strategy_is_registered_by_builder() -> None:
    """The strategy factory should expose cost-latency-pareto under its enum name."""
    strategies = build_strategies(
        default_model_catalog(),
        LatencyStats(),
        InflightStats(),
        0.72,
        "gpt-4.1-mini",
        "claude-haiku-4-5",
        0.5,
        CircuitBreakerRegistry(),
        0.5,
        0.3,
        0.2,
        0.05,
        "gpt-4.1-mini",
        "gpt-5.5",
        0.1,
        750.0,
        min_prefix_chars=64,
        success_stats=SuccessStats(),
    )

    assert isinstance(
        strategies[RoutingStrategyName.COST_LATENCY_PARETO],
        CostLatencyParetoStrategy,
    )
