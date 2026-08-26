"""Tests for thinking-model-prefer routing."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    MOONSHOT_BALANCED_MODEL,
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
    InflightStats,
    LatencyStats,
    SuccessStats,
    ThinkingModelPreferStrategy,
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
        request_id="req-thinking-model-prefer",
        messages=[ChatMessage(content="Solve a hard reasoning task.")],
        metadata=metadata or {},
    )


def _signals(complexity: float = 0.5) -> TaskSignals:
    return TaskSignals(
        complexity_score=complexity,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _strategy(
    *,
    threshold: float = 0.7,
    unavailable: set[str] | None = None,
) -> ThinkingModelPreferStrategy:
    return ThinkingModelPreferStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        complexity_threshold=threshold,
    )


def test_thinking_model_prefer_enum_parses() -> None:
    assert RoutingStrategyName("thinking-model-prefer") is RoutingStrategyName.THINKING_MODEL_PREFER


def test_thinking_model_prefer_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="complexity_threshold must be within"):
        _strategy(threshold=1.5)


def test_thinking_model_prefer_quality_first_below_threshold() -> None:
    decision = _strategy().choose(_request(), _signals(0.69))

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.THINKING_MODEL_PREFER


def test_thinking_model_prefer_prefers_thinking_at_threshold_via_signals() -> None:
    # claude-sonnet-4-6 matches the deterministic "sonnet" thinking token.
    decision = _strategy().choose(_request(), _signals(0.7))

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "thinking-capable" in decision.rationale


def test_thinking_model_prefer_metadata_complexity_overrides_signals() -> None:
    decision = _strategy().choose(
        _request({"complexity_score": 0.95}),
        _signals(0.1),
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "thinking-capable" in decision.rationale


def test_thinking_model_prefer_thinking_models_allowlist() -> None:
    decision = _strategy().choose(
        _request(
            {
                "complexity_score": 0.9,
                "thinking_models": [MOONSHOT_BALANCED_MODEL],
            }
        ),
        _signals(0.9),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "thinking-capable" in decision.rationale


def test_thinking_model_prefer_malformed_metadata_falls_back_to_signals() -> None:
    decision = _strategy().choose(
        _request({"complexity_score": "not-a-number"}),
        _signals(0.2),
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_thinking_model_prefer_clamps_out_of_range_metadata() -> None:
    decision = _strategy().choose(_request({"complexity_score": 2.0}), _signals(0.1))

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "thinking-capable" in decision.rationale


def test_thinking_model_prefer_skips_unhealthy_providers() -> None:
    decision = _strategy(unavailable={"anthropic"}).choose(_request(), _signals(0.9))

    assert decision.provider != "anthropic"


def test_thinking_model_prefer_registered_by_strategy_factory() -> None:
    settings = RouterSettings(thinking_complexity_threshold=0.55)
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
        thinking_complexity_threshold=settings.thinking_complexity_threshold,
    )

    strategy = strategies[RoutingStrategyName.THINKING_MODEL_PREFER]
    assert isinstance(strategy, ThinkingModelPreferStrategy)
    assert strategy._complexity_threshold == 0.55  # noqa: SLF001
    assert RouterSettings().thinking_complexity_threshold == 0.7


def test_thinking_model_prefer_above_threshold_uses_name_heuristic() -> None:
    decision = _strategy().choose(_request(), _signals(0.95))

    assert "sonnet" in decision.chosen_model or decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "0.95" in decision.rationale
