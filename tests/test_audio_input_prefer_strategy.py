"""Tests for audio-input-prefer routing."""

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_PRO_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
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
    AudioInputPreferStrategy,
    InflightStats,
    LatencyStats,
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
        request_id="req-audio-input-prefer",
        messages=[ChatMessage(content="Transcribe this clip.")],
        metadata=metadata or {},
    )


def _strategy(
    unavailable: set[str] | None = None,
    capability_map: dict[str, frozenset[str]] | None = None,
) -> AudioInputPreferStrategy:
    return AudioInputPreferStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        capability_map=capability_map,
    )


def test_audio_input_prefer_enum_parses() -> None:
    assert RoutingStrategyName("audio-input-prefer") is RoutingStrategyName.AUDIO_INPUT_PREFER


def test_audio_input_prefer_quality_first_when_absent() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.AUDIO_INPUT_PREFER


def test_audio_input_prefer_falsy_flags_stay_quality_first() -> None:
    decision = _strategy().choose(
        _request({"requires_audio": False, "audio_input": "no", "audio": 0}),
        _signals(),
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_audio_input_prefer_requires_audio_prefers_capable() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"audio"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"requires_audio": True}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "audio-capable" in decision.rationale


def test_audio_input_prefer_audio_input_alias() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"audio"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"audio_input": "yes"}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_audio_input_prefer_audio_alias() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"audio"})
    decision = _strategy(capability_map=capability_map).choose(_request({"audio": 1}), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_audio_input_prefer_audio_models_allowlist() -> None:
    decision = _strategy(capability_map={}).choose(
        _request(
            {
                "requires_audio": True,
                "audio_models": [MOONSHOT_BALANCED_MODEL],
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "audio-capable" in decision.rationale


def test_audio_input_prefer_name_heuristic_when_map_absent() -> None:
    decision = _strategy(capability_map={}).choose(_request({"requires_audio": True}), _signals())

    # Empty map → name heuristic; gemini matches.
    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "audio-capable" in decision.rationale


def test_audio_input_prefer_respects_model_capabilities_override() -> None:
    catalog = default_model_catalog()
    decision = _strategy(capability_map={model: frozenset({"tools"}) for model in catalog}).choose(
        _request(
            {
                "requires_audio": True,
                "model_capabilities": {MOONSHOT_BALANCED_MODEL: "audio"},
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_audio_input_prefer_known_map_audio_capability() -> None:
    # Default known map marks gpt-5.5 / gemini* as audio-capable; gpt-5.5 leads quality.
    decision = _strategy().choose(_request({"requires_audio": True}), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert "audio-capable" in decision.rationale


def test_audio_input_prefer_skips_unhealthy_providers() -> None:
    decision = _strategy(unavailable={"openai", "google"}).choose(
        _request({"requires_audio": True}), _signals()
    )

    assert decision.provider not in {"openai", "google"}


def test_audio_input_prefer_registered_by_strategy_factory() -> None:
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

    strategy = strategies[RoutingStrategyName.AUDIO_INPUT_PREFER]
    assert isinstance(strategy, AudioInputPreferStrategy)
