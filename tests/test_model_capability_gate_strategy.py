"""Tests for model-capability-gate routing."""

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, GEMINI_FLASH_MODEL, MOONSHOT_BALANCED_MODEL
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
    ModelCapabilityGateStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _request(metadata: dict | None = None) -> RouterRequest:
    return RouterRequest(
        request_id="req-capability-gate",
        messages=[ChatMessage(content="Route with a model capability gate.")],
        metadata=metadata or {},
    )


def _strategy(
    unavailable: set[str] | None = None,
    capability_map: dict[str, frozenset[str]] | None = None,
) -> ModelCapabilityGateStrategy:
    return ModelCapabilityGateStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        capability_map=capability_map,
    )


def test_model_capability_gate_enum_parses() -> None:
    assert RoutingStrategyName("model-capability-gate") is RoutingStrategyName.MODEL_CAPABILITY_GATE


def test_model_capability_gate_no_requirement_is_quality_first() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "no required capabilities declared" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.MODEL_CAPABILITY_GATE


def test_model_capability_gate_filters_by_comma_string() -> None:
    decision = _strategy().choose(
        _request({"required_capabilities": "vision, long_context"}), _signals()
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "satisfied by" in decision.rationale


def test_model_capability_gate_filters_by_list() -> None:
    decision = _strategy().choose(
        _request({"required_capabilities": ["vision", "long_context"]}), _signals()
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_model_capability_gate_excludes_models_missing_a_capability() -> None:
    decision = _strategy().choose(_request({"required_capabilities": "vision"}), _signals())

    # moonshot has no vision capability in the known map, so it must not be chosen.
    assert decision.chosen_model != MOONSHOT_BALANCED_MODEL


def test_model_capability_gate_emergency_retains_when_no_match() -> None:
    decision = _strategy().choose(
        _request({"required_capabilities": "vision,tools,long_context,telepathy"}),
        _signals(),
    )

    assert "emergency" in decision.rationale.lower()
    assert "no candidate satisfies required capabilities" in decision.rationale


def test_model_capability_gate_respects_request_capability_override() -> None:
    decision = _strategy().choose(
        _request(
            {
                "required_capabilities": "fine-tuned",
                "model_capabilities": {MOONSHOT_BALANCED_MODEL: "fine-tuned"},
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_model_capability_gate_respects_circuit_health() -> None:
    decision = _strategy(unavailable={"anthropic"}).choose(
        _request({"required_capabilities": "vision"}), _signals()
    )

    assert decision.provider != "anthropic"


def test_model_capability_gate_custom_capability_map_override() -> None:
    strategy = _strategy(capability_map={GEMINI_FLASH_MODEL: frozenset({"vision"})})
    decision = strategy.choose(_request({"required_capabilities": "vision"}), _signals())

    assert decision.chosen_model == GEMINI_FLASH_MODEL


def test_model_capability_gate_registered_by_strategy_factory() -> None:
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
        success_stats=SuccessStats(),
    )

    strategy = strategies[RoutingStrategyName.MODEL_CAPABILITY_GATE]
    assert isinstance(strategy, ModelCapabilityGateStrategy)
