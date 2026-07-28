"""Tests for the concurrency-cap routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_PRO_MODEL,
    OPENAI_FRONTIER_MODEL,
)
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    ConcurrencyCapStrategy,
    InflightStats,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(
        request_id="req-concurrency-cap",
        messages=[ChatMessage(content="Summarize this incident report.")],
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=96,
    )


def test_concurrency_cap_cold_start_picks_top_quality_model() -> None:
    """With every provider below cap, the highest-quality eligible model wins."""
    strategy = ConcurrencyCapStrategy(default_model_catalog(), InflightStats(), per_provider_cap=8)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.CONCURRENCY_CAP
    assert "load 0/8" in decision.rationale


def test_concurrency_cap_skips_provider_at_cap() -> None:
    """A provider at its live cap should not win primary selection."""
    stats = InflightStats()
    stats.begin("anthropic")
    strategy = ConcurrencyCapStrategy(default_model_catalog(), stats, per_provider_cap=1)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "cap 1" in decision.rationale
    assert ANTHROPIC_SAFETY_MODEL not in decision.fallback_chain[:1]


def test_concurrency_cap_respects_domain_before_cap_filter() -> None:
    """Unsupported providers must not win simply because they are below cap."""
    stats = InflightStats()
    stats.begin("anthropic")
    strategy = ConcurrencyCapStrategy(default_model_catalog(), stats, per_provider_cap=1)

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert DomainTag.MEDICAL in default_model_catalog()[decision.chosen_model].supports_domains


def test_concurrency_cap_all_capped_falls_back_to_least_loaded_provider() -> None:
    """If every eligible provider is capped, choose the least-loaded fallback."""
    stats = InflightStats()
    for provider in ("anthropic", "openai", "google", "moonshot"):
        stats.begin(provider)
    stats.begin("anthropic")
    strategy = ConcurrencyCapStrategy(default_model_catalog(), stats, per_provider_cap=1)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert "every eligible provider at or above cap" in decision.rationale
    assert "openai load 1/1" in decision.rationale


def test_concurrency_cap_rejects_invalid_cap() -> None:
    """A cap below one cannot represent usable provider concurrency."""
    with pytest.raises(ValueError, match="per_provider_cap must be >= 1"):
        ConcurrencyCapStrategy(default_model_catalog(), InflightStats(), per_provider_cap=0)


def test_concurrency_cap_strategy_is_registered_by_builder() -> None:
    """The strategy factory should expose concurrency-cap under its enum name."""
    settings = RouterSettings(concurrency_cap=3)
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

    assert isinstance(strategies[RoutingStrategyName.CONCURRENCY_CAP], ConcurrencyCapStrategy)
