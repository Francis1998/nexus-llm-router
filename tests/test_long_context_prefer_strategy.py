"""Tests for long-context-prefer routing."""

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_PRO_MODEL,
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
    LongContextPreferStrategy,
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
        request_id="req-long-context-prefer",
        messages=[ChatMessage(content="Summarize this long document.")],
        metadata=metadata or {},
    )


def _strategy(
    unavailable: set[str] | None = None,
    context_map: dict[str, int] | None = None,
) -> LongContextPreferStrategy:
    return LongContextPreferStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        context_map=context_map,
    )


def test_long_context_prefer_enum_parses() -> None:
    assert RoutingStrategyName("long-context-prefer") is RoutingStrategyName.LONG_CONTEXT_PREFER


def test_long_context_prefer_quality_first_when_absent() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.LONG_CONTEXT_PREFER


def test_long_context_prefer_falsy_long_context_stays_quality_first() -> None:
    decision = _strategy().choose(_request({"long_context": False}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_long_context_prefer_min_context_tokens_prefers_largest() -> None:
    decision = _strategy().choose(_request({"min_context_tokens": 500_000}), _signals())

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "long-context-capable" in decision.rationale
    assert "context 1000000" in decision.rationale


def test_long_context_prefer_long_context_default_threshold() -> None:
    # Default threshold 100k; all catalog models meet it → rank by context then quality.
    decision = _strategy().choose(_request({"long_context": True}), _signals())

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "long-context-capable" in decision.rationale


def test_long_context_prefer_model_context_windows_override() -> None:
    catalog = default_model_catalog()
    # Force every known window below the threshold so only the metadata override wins.
    context_map = dict.fromkeys(catalog, 32_000)
    decision = _strategy(context_map=context_map).choose(
        _request(
            {
                "min_context_tokens": 250_000,
                "model_context_windows": {MOONSHOT_BALANCED_MODEL: 300_000},
            }
        ),
        _signals(),
    )

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "long-context-capable" in decision.rationale


def test_long_context_prefer_known_map_when_catalog_overridden() -> None:
    catalog = default_model_catalog()
    context_map = dict.fromkeys(catalog, 32_000)
    context_map[GEMINI_PRO_MODEL] = 1_000_000
    decision = _strategy(context_map=context_map).choose(
        _request({"min_context_tokens": 128_000}), _signals()
    )

    assert decision.chosen_model == GEMINI_PRO_MODEL


def test_long_context_prefer_name_heuristic_when_maps_absent() -> None:
    # Empty known map → fall back to catalog context_window, then heuristic.
    decision = _strategy(context_map={}).choose(
        _request({"min_context_tokens": 500_000}), _signals()
    )

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "long-context-capable" in decision.rationale


def test_long_context_prefer_non_positive_min_ignored() -> None:
    decision = _strategy().choose(_request({"min_context_tokens": 0}), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality-first" in decision.rationale


def test_long_context_prefer_skips_unhealthy_providers() -> None:
    decision = _strategy(unavailable={"google"}).choose(
        _request({"min_context_tokens": 500_000}), _signals()
    )

    assert decision.provider != "google"


def test_long_context_prefer_registered_by_strategy_factory() -> None:
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

    strategy = strategies[RoutingStrategyName.LONG_CONTEXT_PREFER]
    assert isinstance(strategy, LongContextPreferStrategy)
