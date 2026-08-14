"""Tests for the carbon-latency-blend routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    CarbonLatencyBlendStrategy,
    InflightStats,
    LatencyStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(metadata: dict[str, str] | None = None, *, region: str | None = None) -> RouterRequest:
    return RouterRequest(
        request_id="req-carbon-latency",
        messages=[ChatMessage(content="route green and fast")],
        metadata=metadata or {},
        region=region,
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=100,
    )


def _latencies(values: dict[str, float]) -> LatencyStats:
    stats = LatencyStats()
    for provider, latency_ms in values.items():
        stats.observe(provider, latency_ms)
    return stats


def test_carbon_latency_blend_enum_parses() -> None:
    assert RoutingStrategyName("carbon-latency-blend") is RoutingStrategyName.CARBON_LATENCY_BLEND


def test_carbon_latency_blend_carbon_only_prefers_lowest_intensity() -> None:
    strategy = CarbonLatencyBlendStrategy(
        default_model_catalog(),
        LatencyStats(),
        carbon_weight=1.0,
        latency_weight=0.0,
    )

    decision = strategy.choose(
        _request(
            {
                "carbon_intensity:anthropic": "100",
                "carbon_intensity:openai": "500",
                "carbon_intensity:google": "400",
                "carbon_intensity:moonshot": "600",
            }
        ),
        _signals(),
    )

    assert decision.provider == "anthropic"
    assert "w_carbon=1.00" in decision.rationale


def test_carbon_latency_blend_latency_only_prefers_fastest_provider() -> None:
    strategy = CarbonLatencyBlendStrategy(
        default_model_catalog(),
        _latencies({"anthropic": 800.0, "openai": 100.0, "google": 400.0, "moonshot": 600.0}),
        carbon_weight=0.0,
        latency_weight=1.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.provider == "openai"
    assert "w_latency=1.00" in decision.rationale
    assert "p95 100.0ms" in decision.rationale


def test_carbon_latency_blend_balances_independent_scores() -> None:
    strategy = CarbonLatencyBlendStrategy(
        default_model_catalog(),
        _latencies({"anthropic": 100.0, "openai": 0.0, "google": 40.0, "moonshot": 100.0}),
        carbon_weight=0.5,
        latency_weight=0.5,
    )

    decision = strategy.choose(
        _request(
            {
                "carbon_intensity:anthropic": "0",
                "carbon_intensity:openai": "100",
                "carbon_intensity:google": "40",
                "carbon_intensity:moonshot": "100",
            }
        ),
        _signals(),
    )

    assert decision.provider == "google"
    assert "score 0.600" in decision.rationale


def test_carbon_latency_blend_provider_region_metadata_takes_precedence() -> None:
    strategy = CarbonLatencyBlendStrategy(
        default_model_catalog(),
        LatencyStats(),
        carbon_weight=1.0,
        latency_weight=0.0,
    )

    decision = strategy.choose(
        _request(
            {
                "carbon_intensity:openai:eu": "50",
                "carbon_intensity:openai": "500",
                "carbon_intensity:anthropic": "100",
                "carbon_intensity:google": "200",
                "carbon_intensity:moonshot": "300",
            },
            region="eu",
        ),
        _signals(),
    )

    assert decision.provider == "openai"
    assert "intensity 50.0" in decision.rationale


def test_carbon_latency_blend_uses_provider_region_default_map() -> None:
    strategy = CarbonLatencyBlendStrategy(
        default_model_catalog(),
        LatencyStats(),
        carbon_weight=1.0,
        latency_weight=0.0,
    )

    decision = strategy.choose(_request(region="eu"), _signals())

    assert decision.provider == "anthropic"
    assert "intensity 180.0" in decision.rationale
    assert "region 'eu'" in decision.rationale


@pytest.mark.parametrize(
    ("carbon_weight", "latency_weight", "message"),
    [(-0.1, 0.5, "carbon_weight"), (0.5, -0.1, "latency_weight")],
)
def test_carbon_latency_blend_rejects_negative_weights(
    carbon_weight: float,
    latency_weight: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CarbonLatencyBlendStrategy(
            default_model_catalog(),
            LatencyStats(),
            carbon_weight=carbon_weight,
            latency_weight=latency_weight,
        )


def test_carbon_latency_blend_registered_by_strategy_factory() -> None:
    settings = RouterSettings(
        carbon_latency_carbon_weight=0.7,
        carbon_latency_latency_weight=0.3,
    )
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
        carbon_latency_carbon_weight=settings.carbon_latency_carbon_weight,
        carbon_latency_latency_weight=settings.carbon_latency_latency_weight,
    )

    strategy = strategies[RoutingStrategyName.CARBON_LATENCY_BLEND]
    assert isinstance(strategy, CarbonLatencyBlendStrategy)
    assert strategy._carbon_weight == 0.7  # noqa: SLF001
    assert strategy._latency_weight == 0.3  # noqa: SLF001
    assert RouterSettings().carbon_latency_carbon_weight == 0.5
    assert RouterSettings().carbon_latency_latency_weight == 0.5
