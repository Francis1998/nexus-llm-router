"""Tests for multimodal-input-prefer routing."""

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
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
    MultimodalInputPreferStrategy,
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
        request_id="req-multimodal-input-prefer",
        messages=[ChatMessage(content="Describe the attached media.")],
        metadata=metadata or {},
    )


def _strategy(
    unavailable: set[str] | None = None,
    capability_map: dict[str, frozenset[str]] | None = None,
) -> MultimodalInputPreferStrategy:
    return MultimodalInputPreferStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        capability_map=capability_map,
    )


def test_multimodal_input_prefer_enum_parses() -> None:
    assert (
        RoutingStrategyName("multimodal-input-prefer")
        is RoutingStrategyName.MULTIMODAL_INPUT_PREFER
    )


def test_multimodal_input_prefer_quality_first_when_absent() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.MULTIMODAL_INPUT_PREFER


def test_multimodal_input_prefer_falsy_flags_stay_quality_first() -> None:
    decision = _strategy().choose(
        _request({"has_images": False, "has_audio": "no"}),
        _signals(),
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_multimodal_input_prefer_has_images_prefers_vision() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[GEMINI_FLASH_MODEL] = frozenset({"vision"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"has_images": True}), _signals()
    )

    assert decision.chosen_model == GEMINI_FLASH_MODEL
    assert "multimodal-capable" in decision.rationale


def test_multimodal_input_prefer_has_audio_alias() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"audio"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"has_audio": "yes"}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_multimodal_input_prefer_multimodal_models_allowlist() -> None:
    decision = _strategy(capability_map={}).choose(
        _request(
            {
                "has_images": True,
                "multimodal_models": [MOONSHOT_BALANCED_MODEL],
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "multimodal-capable" in decision.rationale


def test_multimodal_input_prefer_name_heuristic_when_map_absent() -> None:
    decision = _strategy(capability_map={}).choose(_request({"has_images": 1}), _signals())

    # Empty map → name heuristic; claude/gpt-5/gemini match.
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "multimodal-capable" in decision.rationale


def test_multimodal_input_prefer_respects_model_capabilities_override() -> None:
    catalog = default_model_catalog()
    decision = _strategy(capability_map={model: frozenset({"tools"}) for model in catalog}).choose(
        _request(
            {
                "has_images": True,
                "model_capabilities": {MOONSHOT_BALANCED_MODEL: "vision"},
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_multimodal_input_prefer_known_map_vision_capability() -> None:
    # Default known map marks anthropic safety as vision-capable and highest quality.
    decision = _strategy().choose(_request({"has_images": True}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "multimodal-capable" in decision.rationale


def test_multimodal_input_prefer_skips_unhealthy_providers() -> None:
    decision = _strategy(unavailable={"anthropic"}).choose(
        _request({"has_images": True}), _signals()
    )

    assert decision.provider != "anthropic"


def test_multimodal_input_prefer_registered_by_strategy_factory() -> None:
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

    strategy = strategies[RoutingStrategyName.MULTIMODAL_INPUT_PREFER]
    assert isinstance(strategy, MultimodalInputPreferStrategy)
