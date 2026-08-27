"""Tests for streaming-prefer routing."""

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
    StreamingPreferStrategy,
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
        request_id="req-streaming-prefer",
        messages=[ChatMessage(content="Stream the response.")],
        metadata=metadata or {},
    )


def _strategy(
    unavailable: set[str] | None = None,
    capability_map: dict[str, frozenset[str]] | None = None,
) -> StreamingPreferStrategy:
    return StreamingPreferStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        capability_map=capability_map,
    )


def test_streaming_prefer_enum_parses() -> None:
    assert RoutingStrategyName("streaming-prefer") is RoutingStrategyName.STREAMING_PREFER


def test_streaming_prefer_quality_first_when_absent() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.STREAMING_PREFER


def test_streaming_prefer_falsy_flags_stay_quality_first() -> None:
    decision = _strategy().choose(
        _request({"stream": False, "requires_streaming": "no"}),
        _signals(),
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_streaming_prefer_stream_prefers_capable() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"streaming"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"stream": True}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "streaming-capable" in decision.rationale


def test_streaming_prefer_requires_streaming_alias() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"streaming"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"requires_streaming": "yes"}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_streaming_prefer_streaming_models_allowlist() -> None:
    decision = _strategy(capability_map={}).choose(
        _request(
            {
                "stream": True,
                "streaming_models": [MOONSHOT_BALANCED_MODEL],
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "streaming-capable" in decision.rationale


def test_streaming_prefer_name_heuristic_when_map_absent() -> None:
    decision = _strategy(capability_map={}).choose(_request({"stream": 1}), _signals())

    # Empty map → name heuristic; claude/gpt-5/gemini/kimi match.
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "streaming-capable" in decision.rationale


def test_streaming_prefer_respects_model_capabilities_override() -> None:
    catalog = default_model_catalog()
    decision = _strategy(capability_map={model: frozenset({"tools"}) for model in catalog}).choose(
        _request(
            {
                "requires_streaming": True,
                "model_capabilities": {MOONSHOT_BALANCED_MODEL: "streaming"},
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_streaming_prefer_known_map_streaming_capability() -> None:
    # Default known map marks anthropic safety as streaming-capable and highest quality.
    decision = _strategy().choose(_request({"stream": True}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "streaming-capable" in decision.rationale


def test_streaming_prefer_skips_unhealthy_providers() -> None:
    decision = _strategy(unavailable={"anthropic"}).choose(_request({"stream": True}), _signals())

    assert decision.provider != "anthropic"


def test_streaming_prefer_registered_by_strategy_factory() -> None:
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

    strategy = strategies[RoutingStrategyName.STREAMING_PREFER]
    assert isinstance(strategy, StreamingPreferStrategy)
