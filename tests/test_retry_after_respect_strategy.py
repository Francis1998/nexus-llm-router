"""Tests for the retry-after-respect routing strategy."""

from __future__ import annotations

from unittest.mock import patch

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
    ProviderRetryAfterCooldown,
    RetryAfterRespectStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(
        request_id="req-retry-after",
        messages=[ChatMessage(content="Respect Retry-After waits.")],
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
    cooldown: ProviderRetryAfterCooldown | None = None,
    health: CircuitBreakerRegistry | None = None,
) -> RetryAfterRespectStrategy:
    """Build retry-after-respect with overridable dependencies."""
    return RetryAfterRespectStrategy(
        default_model_catalog(),
        health or CircuitBreakerRegistry(),
        cooldown or ProviderRetryAfterCooldown(30.0),
    )


def test_retry_after_respect_enum_parses() -> None:
    """The API header parser can resolve the strategy value."""
    assert RoutingStrategyName("retry-after-respect") is RoutingStrategyName.RETRY_AFTER_RESPECT


def test_retry_after_cooldown_rejects_negative_default() -> None:
    """Default Retry-After seconds must be non-negative."""
    with pytest.raises(ValueError, match="default_seconds"):
        ProviderRetryAfterCooldown(-1.0)


def test_retry_after_respect_skips_cooling_provider() -> None:
    """A cooling top-quality provider should lose to the next healthy ready one."""
    cooldown = ProviderRetryAfterCooldown(30.0)
    with patch("router.strategies.time.monotonic", return_value=100.0):
        cooldown.set_cooldown("anthropic", 30.0)
        strategy = _strategy(cooldown=cooldown)
        decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert decision.routing_strategy is RoutingStrategyName.RETRY_AFTER_RESPECT
    assert "healthy ready provider openai" in decision.rationale


def test_retry_after_respect_expires_cooldown() -> None:
    """After the wait expires the original provider becomes eligible again."""
    cooldown = ProviderRetryAfterCooldown(30.0)
    clock = {"now": 100.0}

    def _now() -> float:
        return clock["now"]

    with patch("router.strategies.time.monotonic", side_effect=_now):
        cooldown.set_cooldown("anthropic", 30.0)
        clock["now"] = 140.0
        assert cooldown.is_cooling_down("anthropic") is False
        strategy = _strategy(cooldown=cooldown)
        decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "healthy ready provider anthropic" in decision.rationale


def test_retry_after_respect_falls_back_to_next_healthy_when_all_cooling() -> None:
    """When every healthy provider is cooling, pick the soonest remaining wait."""
    cooldown = ProviderRetryAfterCooldown(30.0)
    health = CircuitBreakerRegistry()
    with patch("router.strategies.time.monotonic", return_value=50.0):
        cooldown.set_cooldown("anthropic", 40.0)
        cooldown.set_cooldown("openai", 10.0)
        cooldown.set_cooldown("google", 20.0)
        cooldown.set_cooldown("moonshot", 30.0)
        strategy = _strategy(cooldown=cooldown, health=health)
        decision = strategy.choose(_request(), _signals())

    assert decision.provider == "openai"
    assert "all healthy providers cooling" in decision.rationale
    assert "10.0s remaining" in decision.rationale


def test_retry_after_respect_uses_default_seconds_property() -> None:
    """Cooldown map exposes the configured default wait."""
    cooldown = ProviderRetryAfterCooldown(45.0)
    assert cooldown.default_seconds == 45.0
    with patch("router.strategies.time.monotonic", return_value=0.0):
        cooldown.set_cooldown("openai")
        assert cooldown.remaining_seconds("openai") == pytest.approx(45.0)


def test_retry_after_respect_registered_by_strategy_factory() -> None:
    """The built-in strategy map exposes retry-after-respect."""
    settings = RouterSettings(retry_after_default_seconds=15.0)
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
        retry_after_default_seconds=settings.retry_after_default_seconds,
    )
    strategy = strategies[RoutingStrategyName.RETRY_AFTER_RESPECT]
    assert isinstance(strategy, RetryAfterRespectStrategy)
    assert strategy._retry_after_cooldown.default_seconds == 15.0  # noqa: SLF001


def test_retry_after_respect_settings_default() -> None:
    """RouterSettings expose the default Retry-After wait."""
    assert RouterSettings().retry_after_default_seconds == 30.0
