"""Tests for the geo-region routing strategy."""

from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    MOONSHOT_BALANCED_MODEL,
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
from router.strategies import GeoRegionStrategy


def _request(region: str | None = None) -> RouterRequest:
    """Build a minimal router request with an optional region."""
    return RouterRequest(
        request_id="req-geo",
        messages=[ChatMessage(content="Hello")],
        region=region,
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for a domain."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=64,
    )


def test_geo_region_prefers_matching_region_model() -> None:
    """A CN request should prefer Kimi K2 over US-only OpenAI frontier models."""
    strategy = GeoRegionStrategy(default_model_catalog())

    decision = strategy.choose(_request(region="cn"), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert decision.routing_strategy is RoutingStrategyName.GEO_REGION
    assert "cn" in decision.rationale


def test_geo_region_picks_highest_quality_among_region_matches() -> None:
    """Among EU-capable models, the highest quality (Claude Sonnet 4.6) wins."""
    strategy = GeoRegionStrategy(default_model_catalog())

    decision = strategy.choose(_request(region="eu"), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    # OpenAI frontier is US/global only and must not win an EU affinity route.
    assert decision.chosen_model != OPENAI_FRONTIER_MODEL


def test_geo_region_defaults_to_global_when_region_omitted() -> None:
    """Omitting region treats the request as global and picks top quality."""
    strategy = GeoRegionStrategy(default_model_catalog())

    decision = strategy.choose(_request(region=None), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "global" in decision.rationale


def test_geo_region_falls_back_when_no_region_match() -> None:
    """An unknown region falls back to the highest-quality eligible model."""
    strategy = GeoRegionStrategy(default_model_catalog())

    decision = strategy.choose(_request(region="antarctica"), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "no model for region" in decision.rationale


def test_geo_region_respects_domain_eligibility() -> None:
    """Region matching still requires domain support."""
    catalog = {
        "eu-general": ModelCandidate(
            model="eu-general",
            provider="anthropic",
            quality_score=0.99,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
            supported_regions={"eu"},
        ),
        "eu-medical": ModelCandidate(
            model="eu-medical",
            provider="google",
            quality_score=0.90,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.MEDICAL},
            supported_regions={"eu"},
        ),
    }
    strategy = GeoRegionStrategy(catalog)

    decision = strategy.choose(_request(region="eu"), _signals(DomainTag.MEDICAL))

    assert decision.chosen_model == "eu-medical"
