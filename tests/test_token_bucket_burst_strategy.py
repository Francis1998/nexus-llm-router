"""Tests for the token-bucket-burst routing strategy."""

from unittest.mock import patch

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
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
    InflightStats,
    LatencyStats,
    SuccessStats,
    TokenBucketBurstStrategy,
    TokenBucketStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(request_id: str = "req-token-bucket-burst") -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(
        request_id=request_id,
        messages=[ChatMessage(content="Summarize the rollout and next steps.")],
        strategy=RoutingStrategyName.TOKEN_BUCKET_BURST,
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for token-bucket-burst tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def test_token_bucket_stats_rejects_invalid_configuration() -> None:
    """Bucket capacity and refill rate must be positive."""
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        TokenBucketStats(capacity=0, refill_per_second=1.0)
    with pytest.raises(ValueError, match="refill_per_second must be positive"):
        TokenBucketStats(capacity=10, refill_per_second=0.0)


def test_token_bucket_stats_starts_full_and_consumes_tokens() -> None:
    """A fresh provider bucket should start at capacity and decrement on consume."""
    stats = TokenBucketStats(capacity=3, refill_per_second=1.0)
    with patch("router.strategies.time.monotonic", return_value=0.0):
        assert stats.available_tokens("openai") == pytest.approx(3.0)
        stats.consume("openai")
        assert stats.available_tokens("openai") == pytest.approx(2.0)


def test_token_bucket_stats_refills_over_time() -> None:
    """Elapsed time should refill depleted provider buckets up to capacity."""
    stats = TokenBucketStats(capacity=2, refill_per_second=1.0)
    times = iter([0.0, 0.0, 0.0, 0.0, 1.5, 1.5])
    with patch("router.strategies.time.monotonic", side_effect=lambda: next(times)):
        assert stats.available_tokens("anthropic") == pytest.approx(2.0)
        stats.consume("anthropic")
        stats.consume("anthropic")
        assert stats.available_tokens("anthropic") == pytest.approx(0.0)
        assert stats.available_tokens("anthropic") == pytest.approx(1.5)


def test_token_bucket_burst_cold_start_picks_top_quality_model() -> None:
    """With full buckets, the highest-quality eligible model should win."""
    strategy = TokenBucketBurstStrategy(default_model_catalog(), TokenBucketStats(10, 1.0))
    with patch("router.strategies.time.monotonic", return_value=0.0):
        decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.TOKEN_BUCKET_BURST
    assert "10.0/10 tokens" in decision.rationale


def test_token_bucket_burst_skips_depleted_provider() -> None:
    """Providers without burst tokens should lose to peers that still have quota."""
    stats = TokenBucketStats(capacity=1, refill_per_second=0.1)
    with patch("router.strategies.time.monotonic", return_value=0.0):
        stats.consume("anthropic")
        strategy = TokenBucketBurstStrategy(default_model_catalog(), stats)
        decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "tokens" in decision.rationale


def test_token_bucket_burst_all_empty_falls_back_to_highest_fraction() -> None:
    """When every bucket is empty, the provider with the most refill progress wins."""
    stats = TokenBucketStats(capacity=10, refill_per_second=1.0)
    with patch("router.strategies.time.monotonic", return_value=0.0):
        for provider in ("anthropic", "openai", "google", "moonshot"):
            for _ in range(10):
                stats.consume(provider)
        stats._buckets["openai"].tokens = 0.8
        stats._buckets["anthropic"].tokens = 0.2
        strategy = TokenBucketBurstStrategy(default_model_catalog(), stats)
        decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert decision.provider == "openai"
    assert "every eligible provider bucket empty" in decision.rationale
    assert "8.00%" in decision.rationale


def test_token_bucket_burst_all_empty_tie_breaks_on_cost() -> None:
    """Equal empty fractions should fall back to the cheapest eligible model."""
    catalog = {
        "premium-model": ModelCandidate(
            model="premium-model",
            provider="premium-provider",
            quality_score=0.99,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
            supports_domains={DomainTag.GENERAL},
        ),
        "cheap-model": ModelCandidate(
            model="cheap-model",
            provider="cheap-provider",
            quality_score=0.70,
            input_cost_per_1k=0.0001,
            output_cost_per_1k=0.0002,
            supports_domains={DomainTag.GENERAL},
        ),
    }
    stats = TokenBucketStats(capacity=1, refill_per_second=0.1)
    with patch("router.strategies.time.monotonic", return_value=0.0):
        stats.consume("premium-provider")
        stats.consume("cheap-provider")
        strategy = TokenBucketBurstStrategy(catalog, stats)
        decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == "cheap-model"
    assert "lowest cost" in decision.rationale


def test_token_bucket_burst_consumes_token_on_selection() -> None:
    """Choosing a provider should decrement its shared bucket by one token."""
    stats = TokenBucketStats(capacity=2, refill_per_second=1.0)
    strategy = TokenBucketBurstStrategy(default_model_catalog(), stats)
    with patch("router.strategies.time.monotonic", return_value=0.0):
        strategy.choose(_request("req-1"), _signals())
        strategy.choose(_request("req-2"), _signals())
        assert stats.available_tokens("anthropic") == pytest.approx(0.0)


def test_token_bucket_burst_strategy_name_parses_header_value() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("token-bucket-burst") is RoutingStrategyName.TOKEN_BUCKET_BURST


def test_token_bucket_burst_is_registered_by_strategy_builder() -> None:
    """The central strategy factory should expose token-bucket-burst."""
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
        settings.concurrency_cap,
        token_bucket_stats=TokenBucketStats(
            settings.token_bucket_capacity,
            settings.token_bucket_refill_per_sec,
        ),
    )

    assert isinstance(
        strategies[RoutingStrategyName.TOKEN_BUCKET_BURST],
        TokenBucketBurstStrategy,
    )


def test_token_bucket_burst_respects_domain_eligibility() -> None:
    """Unsupported domains must not win even when their provider has more tokens."""
    stats = TokenBucketStats(capacity=5, refill_per_second=1.0)
    stats.consume("openai")
    stats.consume("openai")
    strategy = TokenBucketBurstStrategy(default_model_catalog(), stats)

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert decision.chosen_model in {ANTHROPIC_SAFETY_MODEL, MOONSHOT_BALANCED_MODEL}
    assert DomainTag.MEDICAL in default_model_catalog()[decision.chosen_model].supports_domains
