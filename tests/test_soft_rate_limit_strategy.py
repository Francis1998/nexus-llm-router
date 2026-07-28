"""Tests for the soft-rate-limit routing strategy."""

from pathlib import Path

import pytest

from adapters.base import ProviderError
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from router.config import RouterSettings, default_model_catalog
from router.engine import NexusRouter
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_FRONTIER_MODEL
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    ModelCandidate,
    ProviderResponse,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    InflightStats,
    LatencyStats,
    RateLimitStats,
    SoftRateLimitStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _RateLimitedAdapter(MockProviderAdapter):
    """Mock adapter that raises a rate-limit shaped provider error."""

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
    ) -> ProviderResponse:
        """Raise a 429/rate-limit error for engine signal tests."""
        del model, messages, max_tokens
        raise ProviderError("HTTP 429 rate limit exceeded")


def _request(request_id: str = "req-soft-rate-limit") -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(
        request_id=request_id,
        messages=[ChatMessage(content="Summarize the incident and next steps.")],
        strategy=RoutingStrategyName.SOFT_RATE_LIMIT,
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for soft-rate-limit tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def test_rate_limit_stats_tracks_bounded_recent_observations() -> None:
    """Recent 429 pressure should be counted over a bounded provider window."""
    stats = RateLimitStats(max_observations=3)

    stats.observe("anthropic", rate_limited=True)
    stats.observe("anthropic", rate_limited=False)
    stats.observe("anthropic", rate_limited=True)
    stats.observe("anthropic", rate_limited=True)

    assert stats.rate_limit_count("anthropic") == 2
    assert stats.rate_limit_rate("anthropic") == pytest.approx(2 / 3)
    assert stats.rate_limit_count("openai") == 0
    assert stats.rate_limit_rate("openai") == 0.0


def test_rate_limit_stats_rejects_non_positive_window() -> None:
    """A non-positive observation window cannot age rate-limit signals."""
    with pytest.raises(ValueError, match="max_observations must be positive"):
        RateLimitStats(max_observations=0)


def test_soft_rate_limit_cold_start_picks_top_quality_model() -> None:
    """With no rate-limit observations, the top-quality healthy provider wins."""
    strategy = SoftRateLimitStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        RateLimitStats(),
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.SOFT_RATE_LIMIT
    assert "0 recent rate-limit" in decision.rationale


def test_soft_rate_limit_routes_around_recent_429_provider() -> None:
    """A provider with recent 429 pressure should lose to a healthy peer."""
    stats = RateLimitStats()
    stats.observe("anthropic", rate_limited=True)
    strategy = SoftRateLimitStrategy(default_model_catalog(), CircuitBreakerRegistry(), stats)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "0 recent rate-limit" in decision.rationale


def test_soft_rate_limit_prefers_healthy_provider_before_rate_limit_score() -> None:
    """Open-circuit providers should not win solely because they have fewer 429s."""
    catalog = {
        "healthy-model": ModelCandidate(
            model="healthy-model",
            provider="healthy-provider",
            quality_score=0.90,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
        ),
        "open-circuit-model": ModelCandidate(
            model="open-circuit-model",
            provider="open-circuit-provider",
            quality_score=0.99,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
        ),
    }
    health = CircuitBreakerRegistry(failure_threshold=1, recovery_window_seconds=60.0)
    health.record_failure("open-circuit-provider")
    stats = RateLimitStats()
    stats.observe("healthy-provider", rate_limited=True)
    strategy = SoftRateLimitStrategy(catalog, health, stats)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == "healthy-model"
    assert decision.fallback_chain == ["open-circuit-model"]
    assert "healthy providers considered first" in decision.rationale


def test_soft_rate_limit_strategy_name_parses_header_value() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("soft-rate-limit") is RoutingStrategyName.SOFT_RATE_LIMIT


def test_soft_rate_limit_is_registered_by_strategy_builder() -> None:
    """The central strategy factory should expose soft-rate-limit."""
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
        rate_limit_stats=RateLimitStats(),
    )

    assert isinstance(
        strategies[RoutingStrategyName.SOFT_RATE_LIMIT],
        SoftRateLimitStrategy,
    )


@pytest.mark.asyncio
async def test_engine_records_rate_limit_signal_and_next_request_avoids_provider(
    tmp_path: Path,
) -> None:
    """Engine 429 detection should feed the next soft-rate-limit decision."""
    router = NexusRouter(
        settings=RouterSettings(audit_log_path=str(tmp_path / "audit.jsonl")),
        adapter_registry=AdapterRegistry(
            {
                "openai": MockProviderAdapter("openai"),
                "anthropic": _RateLimitedAdapter("anthropic"),
                "google": MockProviderAdapter("google"),
                "moonshot": MockProviderAdapter("moonshot"),
            },
        ),
    )

    first_response = await router.complete(_request("req-soft-rate-limit-1"))
    second_response = await router.complete(_request("req-soft-rate-limit-2"))

    assert first_response.model_used == OPENAI_FRONTIER_MODEL
    assert "fallback attempt" in first_response.rationale
    assert router._rate_limit_stats.rate_limit_count("anthropic") == 1
    assert second_response.model_used == OPENAI_FRONTIER_MODEL
    assert "provider openai" in second_response.rationale
