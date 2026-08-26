"""Tests for tool-calling-prefer routing."""

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
    ToolCallingPreferStrategy,
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
        request_id="req-tool-calling-prefer",
        messages=[ChatMessage(content="Call a tool if needed.")],
        metadata=metadata or {},
    )


def _strategy(
    unavailable: set[str] | None = None,
    capability_map: dict[str, frozenset[str]] | None = None,
) -> ToolCallingPreferStrategy:
    return ToolCallingPreferStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        capability_map=capability_map,
    )


def test_tool_calling_prefer_enum_parses() -> None:
    assert RoutingStrategyName("tool-calling-prefer") is RoutingStrategyName.TOOL_CALLING_PREFER


def test_tool_calling_prefer_quality_first_when_absent() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.TOOL_CALLING_PREFER


def test_tool_calling_prefer_falsy_requires_tools_stays_quality_first() -> None:
    decision = _strategy().choose(_request({"requires_tools": False}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_tool_calling_prefer_requires_tools_prefers_tool_capable() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"json"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"tools"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"requires_tools": True}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "tool-capable" in decision.rationale


def test_tool_calling_prefer_tools_list_alias() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"json"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"tools"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"tools": [{"name": "search"}]}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_tool_calling_prefer_tool_capable_models_allowlist() -> None:
    decision = _strategy(capability_map={}).choose(
        _request(
            {
                "requires_tools": "yes",
                "tool_capable_models": [MOONSHOT_BALANCED_MODEL],
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "tool-capable" in decision.rationale


def test_tool_calling_prefer_name_heuristic_when_map_absent() -> None:
    decision = _strategy(capability_map={}).choose(_request({"requires_tools": 1}), _signals())

    # Empty map → name heuristic; frontier names match gpt-5/claude/gemini/kimi.
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "tool-capable" in decision.rationale


def test_tool_calling_prefer_respects_model_capabilities_override() -> None:
    catalog = default_model_catalog()
    decision = _strategy(capability_map={model: frozenset({"json"}) for model in catalog}).choose(
        _request(
            {
                "requires_tools": True,
                "model_capabilities": {MOONSHOT_BALANCED_MODEL: "tools"},
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_tool_calling_prefer_empty_tools_list_ignored() -> None:
    decision = _strategy().choose(_request({"tools": []}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_tool_calling_prefer_skips_unhealthy_providers() -> None:
    decision = _strategy(unavailable={"anthropic"}).choose(_request(), _signals())

    assert decision.provider != "anthropic"


def test_tool_calling_prefer_registered_by_strategy_factory() -> None:
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

    strategy = strategies[RoutingStrategyName.TOOL_CALLING_PREFER]
    assert isinstance(strategy, ToolCallingPreferStrategy)
