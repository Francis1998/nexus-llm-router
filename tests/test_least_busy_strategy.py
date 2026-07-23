"""Tests for the least-busy routing strategy."""

from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_PRO_MODEL,
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
    InflightStats,
    LatencyStats,
    LeastBusyStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(request_id="req-least-busy", messages=[ChatMessage(content="hello")])


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=64,
    )


def test_inflight_stats_tracks_provider_load_and_clamps_at_zero() -> None:
    """Provider load scores should reflect live attempts only."""
    stats = InflightStats()

    stats.begin("openai")
    stats.begin("openai")
    stats.finish("openai")
    stats.finish("openai")
    stats.finish("openai")

    assert stats.load_score("openai") == 0
    assert stats.load_score("anthropic") == 0


def test_least_busy_cold_start_picks_top_quality_model() -> None:
    """With equal load everywhere, the highest-quality eligible model wins."""
    strategy = LeastBusyStrategy(default_model_catalog(), InflightStats())

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.LEAST_BUSY
    assert "load 0" in decision.rationale


def test_least_busy_routes_around_busy_high_quality_provider() -> None:
    """A busy quality leader is skipped for the best idle eligible provider."""
    stats = InflightStats()
    stats.begin("anthropic")
    strategy = LeastBusyStrategy(default_model_catalog(), stats)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "load 0" in decision.rationale


def test_least_busy_respects_domain_eligibility_before_load() -> None:
    """Unsupported providers must not win simply because they are idle."""
    stats = InflightStats()
    stats.begin("anthropic")
    strategy = LeastBusyStrategy(default_model_catalog(), stats)

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert DomainTag.MEDICAL in default_model_catalog()[decision.chosen_model].supports_domains


def test_least_busy_ties_by_quality_then_lower_cost() -> None:
    """Load ties prefer higher quality, then lower estimated request cost."""
    catalog = {
        "expensive": ModelCandidate(
            model="expensive",
            provider="expensive-provider",
            quality_score=0.9,
            input_cost_per_1k=0.010,
            output_cost_per_1k=0.020,
            supports_domains={DomainTag.GENERAL},
        ),
        "cheap": ModelCandidate(
            model="cheap",
            provider="cheap-provider",
            quality_score=0.9,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
        ),
        "lower-quality": ModelCandidate(
            model="lower-quality",
            provider="lower-quality-provider",
            quality_score=0.8,
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
            supports_domains={DomainTag.GENERAL},
        ),
    }
    strategy = LeastBusyStrategy(catalog, InflightStats())

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == "cheap"
    assert decision.fallback_chain[:2] == ["expensive", "lower-quality"]


def test_least_busy_strategy_is_registered_by_builder() -> None:
    """The strategy factory should expose least-busy under its enum name."""
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
        success_stats=SuccessStats(),
    )

    assert isinstance(strategies[RoutingStrategyName.LEAST_BUSY], LeastBusyStrategy)
