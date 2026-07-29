"""Tests for the model-tier-rate-limit routing strategy."""

from unittest.mock import patch

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_FAST_MODEL,
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
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
    ModelTier,
    ModelTierRateLimitStrategy,
    RateLimitStats,
    SuccessStats,
    TierRequestStats,
    TokenBucketStats,
    build_strategies,
    infer_model_tier,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(request_id: str = "req-model-tier-rate-limit") -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(
        request_id=request_id,
        messages=[ChatMessage(content="Summarize the incident and next steps.")],
        strategy=RoutingStrategyName.MODEL_TIER_RATE_LIMIT,
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for model-tier-rate-limit tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


@pytest.mark.parametrize(
    ("model_name", "expected_tier"),
    [
        (OPENAI_FRONTIER_MODEL, ModelTier.FRONTIER),
        ("claude-sonnet-4-6", ModelTier.FRONTIER),
        ("claude-opus-4", ModelTier.FRONTIER),
        ("gemini-3.1-pro-preview", ModelTier.FRONTIER),
        ("kimi-k2", ModelTier.FRONTIER),
        ("o3-mini", ModelTier.ECONOMY),
        (OPENAI_BALANCED_MODEL, ModelTier.ECONOMY),
        (GEMINI_FLASH_MODEL, ModelTier.ECONOMY),
        (ANTHROPIC_FAST_MODEL, ModelTier.MID),
        ("gpt-4.1", ModelTier.MID),
        ("gemini-2.5-pro", ModelTier.MID),
    ],
)
def test_infer_model_tier_classifies_catalog_models(
    model_name: str,
    expected_tier: ModelTier,
) -> None:
    """Model-name heuristics should map frontier, mid, and economy SKUs."""
    assert infer_model_tier(model_name) is expected_tier


def test_tier_request_stats_rejects_invalid_configuration() -> None:
    """Rolling window and retention caps must be positive."""
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        TierRequestStats(window_seconds=0.0)
    with pytest.raises(ValueError, match="max_timestamps must be positive"):
        TierRequestStats(max_timestamps=0)


def test_tier_request_stats_prunes_old_timestamps() -> None:
    """Requests outside the rolling window should not count toward RPM."""
    stats = TierRequestStats(window_seconds=60.0)
    with patch("router.strategies.time.monotonic", side_effect=[0.0, 0.0, 70.0, 70.0]):
        stats.record("openai", now=0.0)
        assert stats.request_count("openai", now=0.0) == 1
        assert stats.request_count("openai", now=70.0) == 0


def test_model_tier_rate_limit_cold_start_picks_top_quality_model() -> None:
    """With empty windows, the highest-quality eligible model should win."""
    strategy = ModelTierRateLimitStrategy(
        default_model_catalog(),
        TierRequestStats(),
        frontier_rpm=30,
        mid_rpm=60,
        economy_rpm=120,
    )
    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.MODEL_TIER_RATE_LIMIT
    assert "frontier tier" in decision.rationale


def test_model_tier_rate_limit_skips_saturated_frontier_provider() -> None:
    """A saturated frontier provider should lose to a healthy peer."""
    stats = TierRequestStats()
    with patch("router.strategies.time.monotonic", return_value=0.0):
        for _ in range(30):
            stats.record("anthropic", now=0.0)
        strategy = ModelTierRateLimitStrategy(
            default_model_catalog(),
            stats,
            frontier_rpm=30,
            mid_rpm=60,
            economy_rpm=120,
        )
        decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "under frontier tier RPM 30" in decision.rationale


def test_model_tier_rate_limit_all_saturated_falls_back_to_least_saturated() -> None:
    """When every provider is saturated, the least-saturated provider should win."""
    catalog = {
        OPENAI_FRONTIER_MODEL: default_model_catalog()[OPENAI_FRONTIER_MODEL],
        ANTHROPIC_SAFETY_MODEL: default_model_catalog()[ANTHROPIC_SAFETY_MODEL],
        "gemini-3.1-pro-preview": default_model_catalog()["gemini-3.1-pro-preview"],
    }
    stats = TierRequestStats()
    with patch("router.strategies.time.monotonic", return_value=0.0):
        for provider in ("anthropic", "openai", "google"):
            for _ in range(30):
                stats.record(provider, now=0.0)
        for _ in range(4):
            stats.record("anthropic", now=0.0)
        for _ in range(9):
            stats.record("google", now=0.0)
        strategy = ModelTierRateLimitStrategy(
            catalog,
            stats,
            frontier_rpm=30,
            mid_rpm=60,
            economy_rpm=120,
        )
        decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert "least-saturated" in decision.rationale


def test_model_tier_rate_limit_economy_tier_allows_higher_provider_rpm() -> None:
    """Economy-tier models should tolerate a higher per-provider RPM ceiling."""
    catalog = {
        "frontier-model": ModelCandidate(
            model="gpt-5.5",
            provider="frontier-provider",
            quality_score=0.99,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
            supports_domains={DomainTag.GENERAL},
        ),
        "economy-model": ModelCandidate(
            model="gpt-4.1-mini",
            provider="economy-provider",
            quality_score=0.70,
            input_cost_per_1k=0.0001,
            output_cost_per_1k=0.0002,
            supports_domains={DomainTag.GENERAL},
        ),
    }
    stats = TierRequestStats()
    with patch("router.strategies.time.monotonic", return_value=0.0):
        for _ in range(30):
            stats.record("frontier-provider", now=0.0)
        for _ in range(50):
            stats.record("economy-provider", now=0.0)
        strategy = ModelTierRateLimitStrategy(
            catalog,
            stats,
            frontier_rpm=30,
            mid_rpm=60,
            economy_rpm=120,
        )
        decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == "gpt-4.1-mini"
    assert "economy tier" in decision.rationale


def test_model_tier_rate_limit_records_request_on_selection() -> None:
    """Choosing a provider should append a timestamp to its rolling window."""
    stats = TierRequestStats()
    strategy = ModelTierRateLimitStrategy(
        default_model_catalog(),
        stats,
        frontier_rpm=30,
        mid_rpm=60,
        economy_rpm=120,
    )
    with patch("router.strategies.time.monotonic", return_value=0.0):
        first = strategy.choose(_request("req-1"), _signals())
        second = strategy.choose(_request("req-2"), _signals())
        assert stats.request_count(first.provider, now=0.0) == 1
        assert stats.request_count(second.provider, now=0.0) == 1


def test_model_tier_rate_limit_strategy_name_parses_header_value() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("model-tier-rate-limit") is RoutingStrategyName.MODEL_TIER_RATE_LIMIT


def test_model_tier_rate_limit_is_registered_by_strategy_builder() -> None:
    """The central strategy factory should expose model-tier-rate-limit."""
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
        rate_limit_stats=RateLimitStats(),
        token_bucket_stats=TokenBucketStats(
            settings.token_bucket_capacity,
            settings.token_bucket_refill_per_sec,
        ),
        tier_request_stats=TierRequestStats(),
        tier_frontier_rpm=settings.tier_frontier_rpm,
        tier_mid_rpm=settings.tier_mid_rpm,
        tier_economy_rpm=settings.tier_economy_rpm,
    )

    assert isinstance(
        strategies[RoutingStrategyName.MODEL_TIER_RATE_LIMIT],
        ModelTierRateLimitStrategy,
    )


def test_model_tier_rate_limit_respects_domain_eligibility() -> None:
    """Unsupported domains must not win even when their provider is less saturated."""
    stats = TierRequestStats()
    with patch("router.strategies.time.monotonic", return_value=0.0):
        for _ in range(20):
            stats.record("openai", now=0.0)
        strategy = ModelTierRateLimitStrategy(
            default_model_catalog(),
            stats,
            frontier_rpm=30,
            mid_rpm=60,
            economy_rpm=120,
        )
        decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert decision.chosen_model in {ANTHROPIC_SAFETY_MODEL}
    assert DomainTag.MEDICAL in default_model_catalog()[decision.chosen_model].supports_domains
