"""Tests for deep-research-prefer routing."""

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    MOONSHOT_BALANCED_MODEL,
)
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    ModelCandidate,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    DeepResearchPreferStrategy,
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
        request_id="req-deep-research-prefer",
        messages=[ChatMessage(content="Route with deep_research capability.")],
        metadata=metadata or {},
    )


def _strategy(
    unavailable: set[str] | None = None,
    capability_map: dict[str, frozenset[str]] | None = None,
    catalog: dict | None = None,
) -> DeepResearchPreferStrategy:
    return DeepResearchPreferStrategy(
        catalog or default_model_catalog(),
        _FakeHealth(unavailable),
        capability_map=capability_map,
    )


def test_deep_research_prefer_enum_parses() -> None:
    assert (
        RoutingStrategyName("deep-research-prefer")
        is RoutingStrategyName.DEEP_RESEARCH_PREFER
    )


def test_deep_research_prefer_quality_first_when_absent() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.DEEP_RESEARCH_PREFER


def test_deep_research_prefer_falsy_flags_stay_quality_first() -> None:
    decision = _strategy().choose(
        _request(
            {
                "requires_deep_research": False,
                "deep_research": "no",
                "o_deep_research": 0,
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_deep_research_prefer_requires_deep_research_prefers_capable() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"deep_research"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"requires_deep_research": True}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "deep_research-capable" in decision.rationale


def test_deep_research_prefer_deep_research_alias() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"deep_research"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"deep_research": "yes"}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_deep_research_prefer_o_deep_research_alias() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"deep_research"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"o_deep_research": True}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_deep_research_prefer_deep_research_models_allowlist() -> None:
    decision = _strategy(capability_map={}).choose(
        _request(
            {
                "requires_deep_research": True,
                "deep_research_models": [MOONSHOT_BALANCED_MODEL],
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "deep_research-capable" in decision.rationale


def test_deep_research_prefer_name_heuristic_when_map_absent() -> None:
    domains = set(DomainTag)
    catalog = {
        "quality-leader": ModelCandidate(
            model="quality-leader",
            provider="anthropic",
            quality_score=0.99,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
            supports_domains=domains,
        ),
        "deep_research-eval": ModelCandidate(
            model="deep_research-eval",
            provider="openai",
            quality_score=0.5,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
            supports_domains=domains,
        ),
    }
    decision = _strategy(capability_map={}, catalog=catalog).choose(
        _request({"requires_deep_research": True}), _signals()
    )

    assert decision.chosen_model == "deep_research-eval"
    assert "deep_research-capable" in decision.rationale


def test_deep_research_prefer_respects_model_capabilities_override() -> None:
    catalog = default_model_catalog()
    decision = _strategy(capability_map={model: frozenset({"tools"}) for model in catalog}).choose(
        _request(
            {
                "requires_deep_research": True,
                "model_capabilities": {MOONSHOT_BALANCED_MODEL: "deep_research"},
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_deep_research_prefer_capability_map_deep_research() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"deep_research"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"requires_deep_research": True}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "deep_research-capable" in decision.rationale


def test_deep_research_prefer_skips_unhealthy_providers() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"deep_research"}) for model in catalog}
    decision = _strategy(unavailable={"moonshot"}, capability_map=capability_map).choose(
        _request({"requires_deep_research": True}), _signals()
    )

    assert decision.provider != "moonshot"


def test_deep_research_prefer_known_model_has_capability() -> None:
    decision = _strategy().choose(_request({"requires_deep_research": True}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "deep_research-capable" in decision.rationale


def test_deep_research_prefer_registered_by_strategy_factory() -> None:
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

    strategy = strategies[RoutingStrategyName.DEEP_RESEARCH_PREFER]
    assert isinstance(strategy, DeepResearchPreferStrategy)
