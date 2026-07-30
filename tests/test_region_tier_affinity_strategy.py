"""Tests for the region-tier-affinity routing strategy."""

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_FAST_MODEL,
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
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
    ModelTier,
    RateLimitStats,
    RegionTierAffinityStrategy,
    SuccessStats,
    TierRequestStats,
    TokenBucketStats,
    build_strategies,
    infer_target_tier,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(region: str | None = None) -> RouterRequest:
    """Build a minimal router request with an optional region."""
    return RouterRequest(
        request_id="req-region-tier",
        messages=[ChatMessage(content="Hello")],
        region=region,
    )


def _signals(
    complexity_score: float,
    domain_tag: DomainTag = DomainTag.GENERAL,
) -> TaskSignals:
    """Build task signals with a complexity score and domain."""
    return TaskSignals(
        complexity_score=complexity_score,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=64,
    )


def test_infer_target_tier_maps_complexity_bands() -> None:
    """Complexity bands map onto frontier, mid, and economy target tiers."""
    assert infer_target_tier(0.9) is ModelTier.FRONTIER
    assert infer_target_tier(0.7) is ModelTier.FRONTIER
    assert infer_target_tier(0.5) is ModelTier.MID
    assert infer_target_tier(0.35) is ModelTier.MID
    assert infer_target_tier(0.1) is ModelTier.ECONOMY


def test_region_tier_affinity_prefers_region_and_tier_match() -> None:
    """A CN frontier request should prefer Kimi K2 over US-only frontier models."""
    strategy = RegionTierAffinityStrategy(default_model_catalog())

    decision = strategy.choose(_request(region="cn"), _signals(0.9))

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert decision.routing_strategy is RoutingStrategyName.REGION_TIER_AFFINITY
    assert "cn" in decision.rationale
    assert "frontier" in decision.rationale


def test_region_tier_affinity_picks_highest_quality_among_both_matches() -> None:
    """Among EU frontier models, Claude Sonnet 4.6 should beat Gemini 3.x."""
    strategy = RegionTierAffinityStrategy(default_model_catalog())

    decision = strategy.choose(_request(region="eu"), _signals(0.9))

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.chosen_model != OPENAI_FRONTIER_MODEL
    assert "region" in decision.rationale and "tier" in decision.rationale


def test_region_tier_affinity_economy_tier_in_region() -> None:
    """A low-complexity EU request should prefer an EU-capable economy model."""
    strategy = RegionTierAffinityStrategy(default_model_catalog())

    decision = strategy.choose(_request(region="eu"), _signals(0.1))

    assert decision.chosen_model == GEMINI_FLASH_MODEL
    assert "economy" in decision.rationale


def test_region_tier_affinity_mid_tier_in_region() -> None:
    """A mid-complexity EU request should prefer Claude Haiku over frontier."""
    strategy = RegionTierAffinityStrategy(default_model_catalog())

    decision = strategy.choose(_request(region="eu"), _signals(0.5))

    assert decision.chosen_model == ANTHROPIC_FAST_MODEL
    assert "mid" in decision.rationale


def test_region_tier_affinity_falls_back_to_tier_when_region_missing() -> None:
    """An unknown region with frontier complexity prefers any frontier model."""
    strategy = RegionTierAffinityStrategy(default_model_catalog())

    decision = strategy.choose(_request(region="antarctica"), _signals(0.9))

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "no region+tier match" in decision.rationale
    assert "frontier" in decision.rationale


def test_region_tier_affinity_falls_back_to_region_when_tier_missing() -> None:
    """When no tier match exists, prefer the highest-quality region match."""
    catalog = {
        "gpt-5.5-eu": ModelCandidate(
            model="gpt-5.5-eu",
            provider="anthropic",
            quality_score=0.90,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            supports_domains={DomainTag.GENERAL},
            supported_regions={"eu"},
        ),
        "gpt-4.1-mini": ModelCandidate(
            model="gpt-4.1-mini",
            provider="openai",
            quality_score=0.84,
            input_cost_per_1k=0.0002,
            output_cost_per_1k=0.0008,
            supports_domains={DomainTag.GENERAL},
            supported_regions={"us"},
        ),
    }
    strategy = RegionTierAffinityStrategy(catalog)

    decision = strategy.choose(_request(region="eu"), _signals(0.5))

    assert decision.chosen_model == "gpt-5.5-eu"
    assert "no mid tier model" in decision.rationale


def test_region_tier_affinity_falls_back_to_quality() -> None:
    """When neither region nor tier matches, fall back to top quality."""
    catalog = {
        "gpt-5.5": ModelCandidate(
            model="gpt-5.5",
            provider="openai",
            quality_score=0.97,
            input_cost_per_1k=0.006,
            output_cost_per_1k=0.018,
            supports_domains={DomainTag.GENERAL},
            supported_regions={"us"},
        ),
        "claude-haiku-4-5": ModelCandidate(
            model="claude-haiku-4-5",
            provider="anthropic",
            quality_score=0.82,
            input_cost_per_1k=0.0008,
            output_cost_per_1k=0.004,
            supports_domains={DomainTag.GENERAL},
            supported_regions={"us"},
        ),
    }
    strategy = RegionTierAffinityStrategy(catalog)

    decision = strategy.choose(_request(region="eu"), _signals(0.1))

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert "fell back to highest-quality" in decision.rationale


def test_region_tier_affinity_defaults_to_global_when_region_omitted() -> None:
    """Omitting region treats the request as global."""
    strategy = RegionTierAffinityStrategy(default_model_catalog())

    decision = strategy.choose(_request(region=None), _signals(0.9))

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "global" in decision.rationale


def test_region_tier_affinity_respects_domain_eligibility() -> None:
    """Region+tier matching still requires domain support."""
    catalog = {
        "claude-sonnet-4-6-general": ModelCandidate(
            model="claude-sonnet-4-6-general",
            provider="anthropic",
            quality_score=0.99,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            supports_domains={DomainTag.GENERAL},
            supported_regions={"eu"},
        ),
        "claude-sonnet-4-6-medical": ModelCandidate(
            model="claude-sonnet-4-6-medical",
            provider="google",
            quality_score=0.90,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            supports_domains={DomainTag.MEDICAL},
            supported_regions={"eu"},
        ),
    }
    strategy = RegionTierAffinityStrategy(catalog)

    decision = strategy.choose(_request(region="eu"), _signals(0.9, DomainTag.MEDICAL))

    assert decision.chosen_model == "claude-sonnet-4-6-medical"


def test_region_tier_affinity_us_economy_prefers_openai_mini() -> None:
    """A low-complexity US request prefers gpt-4.1-mini among US economy SKUs."""
    strategy = RegionTierAffinityStrategy(default_model_catalog())

    decision = strategy.choose(_request(region="us"), _signals(0.1))

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "economy" in decision.rationale


def test_region_tier_affinity_strategy_name_parses_header_value() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("region-tier-affinity") is RoutingStrategyName.REGION_TIER_AFFINITY


def test_region_tier_affinity_is_registered_by_strategy_builder() -> None:
    """The central strategy factory should expose region-tier-affinity."""
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
        hcl_health_weight=settings.hcl_health_weight,
        hcl_cost_weight=settings.hcl_cost_weight,
        hcl_latency_weight=settings.hcl_latency_weight,
        tier_request_stats=TierRequestStats(),
        tier_frontier_rpm=settings.tier_frontier_rpm,
        tier_mid_rpm=settings.tier_mid_rpm,
        tier_economy_rpm=settings.tier_economy_rpm,
    )

    assert isinstance(
        strategies[RoutingStrategyName.REGION_TIER_AFFINITY],
        RegionTierAffinityStrategy,
    )
