"""Tests for the shadow-traffic-mirror routing strategy."""

from hashlib import sha256

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_BALANCED_MODEL
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
    ShadowTrafficMirrorStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(request_id: str = "req-shadow") -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(request_id=request_id, messages=[ChatMessage(content="hello")])


def _signals() -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def _bucket(request_id: str) -> float:
    """Reproduce the strategy's shadow bucket for a request id."""
    digest = sha256(request_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _strategy(shadow_traffic_percent: float = 100.0) -> ShadowTrafficMirrorStrategy:
    """Build a shadow-traffic-mirror strategy for tests."""
    return ShadowTrafficMirrorStrategy(
        default_model_catalog(),
        quality_floor=0.72,
        shadow_traffic_percent=shadow_traffic_percent,
    )


def test_shadow_traffic_mirror_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("shadow-traffic-mirror") is RoutingStrategyName.SHADOW_TRAFFIC_MIRROR


def test_shadow_traffic_mirror_picks_cost_optimal_primary() -> None:
    """Primary selection should mirror cost-optimal under the quality floor."""
    strategy = _strategy(shadow_traffic_percent=0.0)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert decision.routing_strategy is RoutingStrategyName.SHADOW_TRAFFIC_MIRROR
    assert "selected primary" in decision.rationale


def test_shadow_traffic_mirror_annotates_shadow_on_slice() -> None:
    """Shadow slice traffic should annotate a different-provider mirror model."""
    strategy = _strategy(shadow_traffic_percent=100.0)

    decision = strategy.choose(_request("req-shadow-annotate"), _signals())

    primary = default_model_catalog()[decision.chosen_model]
    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "shadow mirror" in decision.rationale
    assert ANTHROPIC_SAFETY_MODEL in decision.rationale
    mirror = default_model_catalog()[ANTHROPIC_SAFETY_MODEL]
    assert mirror.provider != primary.provider
    assert "dual-run telemetry" in decision.rationale


def test_shadow_traffic_mirror_skips_annotation_off_slice() -> None:
    """Off-slice traffic should not queue the shadow mirror."""
    strategy = _strategy(shadow_traffic_percent=0.0)

    decision = strategy.choose(_request("req-shadow-off"), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "not annotated" in decision.rationale
    assert "dual-run telemetry" not in decision.rationale


def test_shadow_traffic_mirror_bucket_is_deterministic() -> None:
    """The same request id should always land in the same shadow slice."""
    request_id = "req-deterministic-shadow"
    bucket = _bucket(request_id)
    strategy = _strategy(shadow_traffic_percent=bucket * 100.0 + 0.01)

    first = strategy.choose(_request(request_id), _signals())
    second = strategy.choose(_request(request_id), _signals())

    assert "dual-run telemetry" in first.rationale
    assert "dual-run telemetry" in second.rationale


def test_shadow_traffic_mirror_rejects_invalid_percent() -> None:
    """Shadow traffic percent must stay inside [0.0, 100.0]."""
    with pytest.raises(ValueError, match="shadow_traffic_percent"):
        _strategy(shadow_traffic_percent=150.0)


def test_shadow_traffic_mirror_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose shadow-traffic-mirror."""
    catalog = default_model_catalog()
    settings = RouterSettings(shadow_traffic_percent=5.0)
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
        shadow_traffic_percent=settings.shadow_traffic_percent,
    )

    strategy = strategies[RoutingStrategyName.SHADOW_TRAFFIC_MIRROR]
    assert isinstance(strategy, ShadowTrafficMirrorStrategy)
    assert strategy.strategy_name is RoutingStrategyName.SHADOW_TRAFFIC_MIRROR
