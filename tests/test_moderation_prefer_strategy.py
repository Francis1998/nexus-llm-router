"""Tests for moderation-prefer routing."""

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
    InflightStats,
    LatencyStats,
    ModerationPreferStrategy,
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
        request_id="req-moderation-prefer",
        messages=[ChatMessage(content="Click the submit button.")],
        metadata=metadata or {},
    )


def _strategy(
    unavailable: set[str] | None = None,
    capability_map: dict[str, frozenset[str]] | None = None,
    catalog: dict | None = None,
) -> ModerationPreferStrategy:
    return ModerationPreferStrategy(
        catalog or default_model_catalog(),
        _FakeHealth(unavailable),
        capability_map=capability_map,
    )


def test_moderation_prefer_enum_parses() -> None:
    assert RoutingStrategyName("moderation-prefer") is RoutingStrategyName.MODERATION_PREFER


def test_moderation_prefer_quality_first_when_absent() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.MODERATION_PREFER


def test_moderation_prefer_falsy_flags_stay_quality_first() -> None:
    decision = _strategy().choose(
        _request(
            {
                "requires_moderation": False,
                "moderation": "no",
                "content_moderation": 0,
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_moderation_prefer_requires_moderation_prefers_capable() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"moderation"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"requires_moderation": True}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "moderation-capable" in decision.rationale


def test_moderation_prefer_moderation_alias() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"moderation"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"moderation": "yes"}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_moderation_prefer_content_moderation_alias() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"moderation"})
    meta = {"content_moderation": True}
    decision = _strategy(capability_map=capability_map).choose(_request(meta), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_moderation_prefer_moderation_models_allowlist() -> None:
    decision = _strategy(capability_map={}).choose(
        _request(
            {
                "requires_moderation": True,
                "moderation_models": [MOONSHOT_BALANCED_MODEL],
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "moderation-capable" in decision.rationale


def test_moderation_prefer_name_heuristic_when_map_absent() -> None:
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
        "omni-moderation-agent": ModelCandidate(
            model="omni-moderation-agent",
            provider="openai",
            quality_score=0.5,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
            supports_domains=domains,
        ),
    }
    decision = _strategy(capability_map={}, catalog=catalog).choose(
        _request({"requires_moderation": True}), _signals()
    )

    assert decision.chosen_model == "omni-moderation-agent"
    assert "moderation-capable" in decision.rationale


def test_moderation_prefer_respects_model_capabilities_override() -> None:
    catalog = default_model_catalog()
    decision = _strategy(capability_map={model: frozenset({"tools"}) for model in catalog}).choose(
        _request(
            {
                "requires_moderation": True,
                "model_capabilities": {MOONSHOT_BALANCED_MODEL: "moderation"},
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_moderation_prefer_capability_map_moderation() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"moderation"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"requires_moderation": True}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "moderation-capable" in decision.rationale


def test_moderation_prefer_skips_unhealthy_providers() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"moderation"}) for model in catalog}
    decision = _strategy(unavailable={"moonshot"}, capability_map=capability_map).choose(
        _request({"requires_moderation": True}), _signals()
    )

    assert decision.provider != "moonshot"


def test_moderation_prefer_registered_by_strategy_factory() -> None:
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

    strategy = strategies[RoutingStrategyName.MODERATION_PREFER]
    assert isinstance(strategy, ModerationPreferStrategy)
