"""Tests for prompt-injection-risk-shed routing."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL
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
    PromptInjectionRiskShedStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _request(metadata: dict | None = None) -> RouterRequest:
    return RouterRequest(
        request_id="req-prompt-injection-risk-shed",
        messages=[ChatMessage(content="Route with prompt injection risk shed.")],
        metadata=metadata or {},
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _strategy(
    *,
    threshold: float = 0.7,
    unavailable: set[str] | None = None,
) -> PromptInjectionRiskShedStrategy:
    return PromptInjectionRiskShedStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        risk_threshold=threshold,
    )


def _cheapest_model() -> str:
    catalog = default_model_catalog()
    cheapest = min(
        catalog.values(),
        key=lambda candidate: candidate.estimate_cost(128, 512),
    )
    return cheapest.model


def test_prompt_injection_risk_shed_enum_parses() -> None:
    assert (
        RoutingStrategyName("prompt-injection-risk-shed")
        is RoutingStrategyName.PROMPT_INJECTION_RISK_SHED
    )


def test_prompt_injection_risk_shed_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="risk_threshold must be within"):
        _strategy(threshold=1.5)


def test_prompt_injection_risk_shed_quality_first_without_metadata() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.PROMPT_INJECTION_RISK_SHED


def test_prompt_injection_risk_shed_quality_first_below_threshold() -> None:
    decision = _strategy().choose(_request({"prompt_injection_risk": 0.69}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_prompt_injection_risk_shed_sheds_at_threshold() -> None:
    decision = _strategy().choose(_request({"prompt_injection_risk": 0.7}), _signals())

    assert decision.chosen_model == _cheapest_model()
    assert "shed to" in decision.rationale
    assert "lowest-cost" in decision.rationale


def test_prompt_injection_risk_shed_sheds_above_threshold() -> None:
    decision = _strategy().choose(_request({"prompt_injection_risk": 0.95}), _signals())

    assert decision.chosen_model == _cheapest_model()


def test_prompt_injection_risk_shed_ignores_malformed_risk() -> None:
    decision = _strategy().choose(_request({"prompt_injection_risk": "not-a-number"}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_prompt_injection_risk_shed_clamps_out_of_range() -> None:
    decision = _strategy(threshold=0.7).choose(_request({"prompt_injection_risk": 2.0}), _signals())

    assert decision.chosen_model == _cheapest_model()


def test_prompt_injection_risk_shed_never_rejects() -> None:
    decision = _strategy().choose(_request({"prompt_injection_risk": 1.0}), _signals())

    assert decision.chosen_model
    assert "reject" not in decision.rationale.lower()


def test_prompt_injection_risk_shed_skips_unhealthy_providers() -> None:
    decision = _strategy(unavailable={"anthropic"}).choose(_request(), _signals())

    assert decision.provider != "anthropic"


def test_prompt_injection_risk_shed_registered_by_strategy_factory() -> None:
    settings = RouterSettings(prompt_injection_risk_threshold=0.55)
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
        success_stats=SuccessStats(),
        prompt_injection_risk_threshold=settings.prompt_injection_risk_threshold,
    )

    strategy = strategies[RoutingStrategyName.PROMPT_INJECTION_RISK_SHED]
    assert isinstance(strategy, PromptInjectionRiskShedStrategy)
    assert strategy._risk_threshold == 0.55  # noqa: SLF001
    assert RouterSettings().prompt_injection_risk_threshold == 0.7
