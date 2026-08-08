"""Tests for the circuit-breaker-half-open-probe routing strategy."""

import time

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
    CircuitBreakerHalfOpenProbeStrategy,
    InflightStats,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(
        request_id="req-half-open-probe",
        messages=[ChatMessage(content="Recover carefully.")],
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=32,
    )


def _open_circuit(health: CircuitBreakerRegistry, provider: str) -> None:
    """Force a provider circuit open using the registry threshold."""
    for _ in range(health._failure_threshold):  # noqa: SLF001
        health.record_failure(provider)


def _force_half_open(health: CircuitBreakerRegistry, provider: str) -> None:
    """Open a provider circuit and rewind opened_at into the probe window."""
    _open_circuit(health, provider)
    state = health._states[provider]  # noqa: SLF001
    assert state.opened_at is not None
    state.opened_at = time.monotonic() - health._recovery_window_seconds - 1.0  # noqa: SLF001


def test_circuit_breaker_half_open_probe_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("circuit-breaker-half-open-probe")
        is RoutingStrategyName.CIRCUIT_BREAKER_HALF_OPEN_PROBE
    )


def test_circuit_breaker_half_open_probe_prefers_healthy_over_half_open() -> None:
    """Healthy closed providers win even when a higher-quality peer is half-open."""
    health = CircuitBreakerRegistry(failure_threshold=1, recovery_window_seconds=60.0)
    _force_half_open(health, "anthropic")
    strategy = CircuitBreakerHalfOpenProbeStrategy(
        default_model_catalog(),
        health,
        InflightStats(),
        probe_budget=2,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "preferred healthy provider" in decision.rationale
    assert health.is_half_open("anthropic") is True
    assert health.is_half_open("openai") is False


def test_circuit_breaker_half_open_probe_allows_probe_under_budget() -> None:
    """With no healthy providers, half-open candidates may receive a probe."""
    health = CircuitBreakerRegistry(failure_threshold=1, recovery_window_seconds=60.0)
    for provider in ("openai", "anthropic", "google", "moonshot"):
        _force_half_open(health, provider)
    strategy = CircuitBreakerHalfOpenProbeStrategy(
        default_model_catalog(),
        health,
        InflightStats(),
        probe_budget=2,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "allowed recovery probe" in decision.rationale
    assert "0/2" in decision.rationale


def test_circuit_breaker_half_open_probe_notes_exhausted_budget() -> None:
    """When half-open in-flight load meets the budget, rationale notes exhaustion."""
    health = CircuitBreakerRegistry(failure_threshold=1, recovery_window_seconds=60.0)
    for provider in ("openai", "anthropic", "google", "moonshot"):
        _force_half_open(health, provider)
    stats = InflightStats()
    stats.begin("anthropic")
    stats.begin("openai")
    strategy = CircuitBreakerHalfOpenProbeStrategy(
        default_model_catalog(),
        health,
        stats,
        probe_budget=2,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "probe budget exhausted" in decision.rationale
    assert "2/2" in decision.rationale


def test_circuit_breaker_half_open_probe_rejects_non_positive_budget() -> None:
    """A non-positive probe budget fails fast at construction."""
    with pytest.raises(ValueError, match=">= 1"):
        CircuitBreakerHalfOpenProbeStrategy(
            default_model_catalog(),
            CircuitBreakerRegistry(),
            InflightStats(),
            probe_budget=0,
        )


def test_circuit_breaker_half_open_probe_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose circuit-breaker-half-open-probe."""
    catalog = default_model_catalog()
    settings = RouterSettings(circuit_half_open_probe_budget=3)
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
        provider_family_cost_ceiling_usd=settings.provider_family_cost_ceiling_usd,
        cache_hit_sticky_min_chars=settings.cache_hit_sticky_min_chars,
        circuit_half_open_probe_budget=settings.circuit_half_open_probe_budget,
    )

    strategy = strategies[RoutingStrategyName.CIRCUIT_BREAKER_HALF_OPEN_PROBE]
    assert isinstance(strategy, CircuitBreakerHalfOpenProbeStrategy)
    assert strategy.strategy_name is RoutingStrategyName.CIRCUIT_BREAKER_HALF_OPEN_PROBE


def test_circuit_breaker_is_half_open_false_while_fully_open() -> None:
    """Providers still inside the recovery window are not half-open."""
    health = CircuitBreakerRegistry(failure_threshold=1, recovery_window_seconds=60.0)
    _open_circuit(health, "openai")

    assert health.is_available("openai") is False
    assert health.is_half_open("openai") is False
