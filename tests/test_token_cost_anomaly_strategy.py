"""Tests for the token-cost-anomaly-shed routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    OPENAI_BALANCED_MODEL,
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
    CostAnomalyStats,
    InflightStats,
    LatencyStats,
    SuccessStats,
    TokenCostAnomalyShedStrategy,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    """Deterministic provider-health view for unit tests."""

    def __init__(self, unavailable: set[str]) -> None:
        """Store providers considered unavailable."""
        self._unavailable = unavailable

    def is_available(self, provider: str) -> bool:
        """Return whether a provider is routable."""
        return provider not in self._unavailable


def _request() -> RouterRequest:
    """Build a minimal router request."""
    return RouterRequest(request_id="req-cost-anomaly", messages=[ChatMessage(content="hello")])


def _signals() -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def _strategy(
    *,
    cost_anomaly_stats: CostAnomalyStats | None = None,
    token_cost_anomaly_ratio: float = 2.0,
    unavailable: set[str] | None = None,
) -> TokenCostAnomalyShedStrategy:
    """Build a token-cost-anomaly-shed strategy for tests."""
    stats = cost_anomaly_stats or CostAnomalyStats()
    return TokenCostAnomalyShedStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable or set()),
        stats,
        token_cost_anomaly_ratio=token_cost_anomaly_ratio,
    )


def test_token_cost_anomaly_shed_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("token-cost-anomaly-shed")
        is RoutingStrategyName.TOKEN_COST_ANOMALY_SHED
    )


def test_token_cost_anomaly_shed_cold_start_picks_top_quality() -> None:
    """With no rolling mean, routing prefers the highest-quality model."""
    strategy = _strategy()

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.TOKEN_COST_ANOMALY_SHED
    assert "highest-quality" in decision.rationale


def test_token_cost_anomaly_shed_sheds_to_cheaper_healthy_when_spike_detected() -> None:
    """An anomalously expensive top pick should shed to a cheaper healthy model."""
    stats = CostAnomalyStats()
    for _ in range(5):
        stats.observe(0.0005)
    strategy = _strategy(cost_anomaly_stats=stats, token_cost_anomaly_ratio=2.0)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "shed to cheaper healthy" in decision.rationale


def test_token_cost_anomaly_shed_quality_fallback_when_no_cheaper_healthy() -> None:
    """When every cheaper option is unhealthy, fall back to top quality."""
    catalog = {
        "premium-model": default_model_catalog()[ANTHROPIC_SAFETY_MODEL].model_copy(
            update={
                "model": "premium-model",
                "provider": "openai",
                "quality_score": 0.99,
                "input_cost_per_1k": 0.02,
                "output_cost_per_1k": 0.06,
            }
        ),
        "economy-model": default_model_catalog()[OPENAI_BALANCED_MODEL].model_copy(
            update={
                "model": "economy-model",
                "provider": "google",
                "quality_score": 0.70,
                "input_cost_per_1k": 0.0001,
                "output_cost_per_1k": 0.0002,
            }
        ),
    }
    stats = CostAnomalyStats()
    for _ in range(5):
        stats.observe(0.0005)
    strategy = TokenCostAnomalyShedStrategy(
        catalog,
        _FakeHealth({"google"}),
        stats,
        token_cost_anomaly_ratio=2.0,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == "premium-model"
    assert "quality fallback" in decision.rationale


def test_token_cost_anomaly_shed_keeps_quality_when_within_baseline() -> None:
    """Top quality should win when its projected cost/1k is within the anomaly band."""
    stats = CostAnomalyStats()
    for _ in range(5):
        stats.observe(0.01)
    strategy = _strategy(cost_anomaly_stats=stats, token_cost_anomaly_ratio=2.0)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "within rolling baseline" in decision.rationale


def test_token_cost_anomaly_shed_rejects_non_positive_ratio() -> None:
    """The anomaly ratio must be strictly positive."""
    with pytest.raises(ValueError, match="token_cost_anomaly_ratio"):
        _strategy(token_cost_anomaly_ratio=0.0)


def test_token_cost_anomaly_shed_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose token-cost-anomaly-shed."""
    catalog = default_model_catalog()
    settings = RouterSettings(token_cost_anomaly_ratio=2.5)
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
        token_cost_anomaly_ratio=settings.token_cost_anomaly_ratio,
    )

    strategy = strategies[RoutingStrategyName.TOKEN_COST_ANOMALY_SHED]
    assert isinstance(strategy, TokenCostAnomalyShedStrategy)
    assert strategy.strategy_name is RoutingStrategyName.TOKEN_COST_ANOMALY_SHED
