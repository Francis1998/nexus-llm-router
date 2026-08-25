"""Tests for provider-exclusion routing."""

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
    InflightStats,
    LatencyStats,
    ProviderExclusionStrategy,
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
        request_id="req-provider-exclusion",
        messages=[ChatMessage(content="Route with provider exclusion.")],
        metadata=metadata or {},
    )


def _strategy(unavailable: set[str] | None = None) -> ProviderExclusionStrategy:
    return ProviderExclusionStrategy(default_model_catalog(), _FakeHealth(unavailable))


def test_provider_exclusion_enum_parses() -> None:
    assert RoutingStrategyName("provider-exclusion") is RoutingStrategyName.PROVIDER_EXCLUSION


def test_provider_exclusion_quality_first_without_exclusions() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_EXCLUSION
    assert "emergency" not in decision.rationale


def test_provider_exclusion_filters_excluded_providers_list() -> None:
    decision = _strategy().choose(_request({"excluded_providers": ["anthropic"]}), _signals())

    assert decision.provider != "anthropic"
    assert decision.chosen_model == OPENAI_FRONTIER_MODEL


def test_provider_exclusion_filters_excluded_providers_csv() -> None:
    decision = _strategy().choose(_request({"excluded_providers": "anthropic, openai"}), _signals())

    assert decision.provider not in {"anthropic", "openai"}
    assert decision.chosen_model == GEMINI_PRO_MODEL


def test_provider_exclusion_filters_excluded_models() -> None:
    decision = _strategy().choose(
        _request({"excluded_models": [ANTHROPIC_SAFETY_MODEL, OPENAI_FRONTIER_MODEL]}),
        _signals(),
    )

    assert decision.chosen_model == GEMINI_PRO_MODEL


def test_provider_exclusion_emergency_when_all_excluded() -> None:
    catalog = default_model_catalog()
    decision = _strategy().choose(
        _request({"excluded_providers": sorted({c.provider for c in catalog.values()})}),
        _signals(),
    )

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "emergency" in decision.rationale.lower()


def test_provider_exclusion_skips_unhealthy_providers() -> None:
    decision = _strategy(unavailable={"anthropic"}).choose(_request(), _signals())

    assert decision.provider != "anthropic"


def test_provider_exclusion_model_and_provider_together() -> None:
    decision = _strategy().choose(
        _request(
            {
                "excluded_providers": "anthropic",
                "excluded_models": OPENAI_FRONTIER_MODEL,
            }
        ),
        _signals(),
    )

    assert decision.provider != "anthropic"
    assert decision.chosen_model != OPENAI_FRONTIER_MODEL
    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert decision.chosen_model != MOONSHOT_BALANCED_MODEL


def test_provider_exclusion_registered_by_strategy_factory() -> None:
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

    strategy = strategies[RoutingStrategyName.PROVIDER_EXCLUSION]
    assert isinstance(strategy, ProviderExclusionStrategy)
