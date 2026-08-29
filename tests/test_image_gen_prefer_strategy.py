"""Tests for image-gen-prefer routing."""

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
    ImageGenPreferStrategy,
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
        request_id="req-image-gen-prefer",
        messages=[ChatMessage(content="Click the submit button.")],
        metadata=metadata or {},
    )


def _strategy(
    unavailable: set[str] | None = None,
    capability_map: dict[str, frozenset[str]] | None = None,
    catalog: dict | None = None,
) -> ImageGenPreferStrategy:
    return ImageGenPreferStrategy(
        catalog or default_model_catalog(),
        _FakeHealth(unavailable),
        capability_map=capability_map,
    )


def test_image_gen_prefer_enum_parses() -> None:
    assert RoutingStrategyName("image-gen-prefer") is RoutingStrategyName.IMAGE_GEN_PREFER


def test_image_gen_prefer_quality_first_when_absent() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.IMAGE_GEN_PREFER


def test_image_gen_prefer_falsy_flags_stay_quality_first() -> None:
    decision = _strategy().choose(
        _request({"requires_image_gen": False, "image_gen": "no", "image_generation": 0}),
        _signals(),
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_image_gen_prefer_requires_image_gen_prefers_capable() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"image_gen"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"requires_image_gen": True}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "image-gen-capable" in decision.rationale


def test_image_gen_prefer_computer_use_alias() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"image_gen"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"image_gen": "yes"}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_image_gen_prefer_cua_alias() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"image_gen"})
    meta = {"image_generation": True}
    decision = _strategy(capability_map=capability_map).choose(_request(meta), _signals())

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_image_gen_prefer_image_gen_models_allowlist() -> None:
    decision = _strategy(capability_map={}).choose(
        _request(
            {
                "requires_image_gen": True,
                "image_gen_models": [MOONSHOT_BALANCED_MODEL],
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "image-gen-capable" in decision.rationale


def test_image_gen_prefer_name_heuristic_when_map_absent() -> None:
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
        "image-gen-agent": ModelCandidate(
            model="image-gen-agent",
            provider="openai",
            quality_score=0.5,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
            supports_domains=domains,
        ),
    }
    decision = _strategy(capability_map={}, catalog=catalog).choose(
        _request({"requires_image_gen": True}), _signals()
    )

    assert decision.chosen_model == "image-gen-agent"
    assert "image-gen-capable" in decision.rationale


def test_image_gen_prefer_respects_model_capabilities_override() -> None:
    catalog = default_model_catalog()
    decision = _strategy(capability_map={model: frozenset({"tools"}) for model in catalog}).choose(
        _request(
            {
                "requires_image_gen": True,
                "model_capabilities": {MOONSHOT_BALANCED_MODEL: "image_gen"},
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_image_gen_prefer_capability_map_computer_use() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"tools"}) for model in catalog}
    capability_map[MOONSHOT_BALANCED_MODEL] = frozenset({"image_gen"})
    decision = _strategy(capability_map=capability_map).choose(
        _request({"requires_image_gen": True}), _signals()
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "image-gen-capable" in decision.rationale


def test_image_gen_prefer_skips_unhealthy_providers() -> None:
    catalog = default_model_catalog()
    capability_map = {model: frozenset({"image_gen"}) for model in catalog}
    decision = _strategy(unavailable={"moonshot"}, capability_map=capability_map).choose(
        _request({"requires_image_gen": True}), _signals()
    )

    assert decision.provider != "moonshot"


def test_image_gen_prefer_registered_by_strategy_factory() -> None:
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

    strategy = strategies[RoutingStrategyName.IMAGE_GEN_PREFER]
    assert isinstance(strategy, ImageGenPreferStrategy)
