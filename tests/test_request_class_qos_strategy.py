"""Tests for request-class-qos routing."""

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, GEMINI_PRO_MODEL
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
    RequestClass,
    RequestClassQosStrategy,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self._unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self._unavailable


def _request(request_class: str | None = None, *, qos_class: str | None = None) -> RouterRequest:
    metadata: dict[str, str] = {}
    if request_class is not None:
        metadata["request_class"] = request_class
    if qos_class is not None:
        metadata["qos_class"] = qos_class
    return RouterRequest(
        request_id="req-qos",
        messages=[ChatMessage(content="Route by request class QoS.")],
        metadata=metadata,
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _latencies() -> LatencyStats:
    stats = LatencyStats()
    stats.observe("anthropic", 500.0)
    stats.observe("google", 50.0)
    stats.observe("moonshot", 100.0)
    stats.observe("openai", 200.0)
    return stats


def _strategy(*, unavailable: set[str] | None = None) -> RequestClassQosStrategy:
    return RequestClassQosStrategy(
        default_model_catalog(),
        _FakeHealth(unavailable),
        _latencies(),
    )


def test_request_class_qos_enum_parses() -> None:
    assert RoutingStrategyName("request-class-qos") is RoutingStrategyName.REQUEST_CLASS_QOS


def test_request_class_qos_interactive_picks_fastest_healthy() -> None:
    decision = _strategy(unavailable={"openai"}).choose(_request("interactive"), _signals())

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert decision.provider == "google"
    assert "interactive" in decision.rationale
    assert "lowest observed latency high-quality route" in decision.rationale


def test_request_class_qos_defaults_to_interactive() -> None:
    decision = _strategy(unavailable={"openai"}).choose(_request(), _signals())

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "interactive" in decision.rationale


def test_request_class_qos_qos_class_alias_works() -> None:
    decision = _strategy(unavailable={"openai"}).choose(
        _request(qos_class="interactive"),
        _signals(),
    )

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "interactive" in decision.rationale


def test_request_class_qos_batch_prefers_quality_mid_cost() -> None:
    decision = _strategy().choose(_request("batch"), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "batch" in decision.rationale
    assert "quality-first mid-cost route" in decision.rationale


def test_request_class_qos_bulk_picks_cheapest_healthy() -> None:
    decision = _strategy(unavailable={"openai"}).choose(_request("bulk"), _signals())

    candidates = [
        candidate
        for candidate in default_model_catalog().values()
        if candidate.provider != "openai" and DomainTag.GENERAL in candidate.supports_domains
    ]
    cheapest = min(
        candidates,
        key=lambda candidate: (
            candidate.estimate_cost(_signals().prompt_tokens_estimate, 512),
            -candidate.quality_score,
            candidate.model,
        ),
    )
    assert decision.chosen_model == cheapest.model
    assert "bulk" in decision.rationale
    assert "cheapest healthy route" in decision.rationale


def test_request_class_qos_unknown_class_falls_back_to_interactive() -> None:
    decision = _strategy(unavailable={"openai"}).choose(_request("urgent"), _signals())

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert "interactive" in decision.rationale


def test_request_class_qos_emergency_when_all_unhealthy() -> None:
    decision = _strategy(
        unavailable={"openai", "anthropic", "google", "moonshot"},
    ).choose(_request("bulk"), _signals())

    assert "emergency" in decision.rationale
    assert decision.routing_strategy is RoutingStrategyName.REQUEST_CLASS_QOS


def test_request_class_enum_values() -> None:
    assert RequestClass.INTERACTIVE.value == "interactive"
    assert RequestClass.BATCH.value == "batch"
    assert RequestClass.BULK.value == "bulk"


def test_request_class_qos_registered_by_strategy_factory() -> None:
    strategies = build_strategies(
        default_model_catalog(),
        LatencyStats(),
        InflightStats(),
        RouterSettings().quality_floor,
        RouterSettings().ab_model_a,
        RouterSettings().ab_model_b,
        RouterSettings().ab_model_a_weight,
        CircuitBreakerRegistry(),
        RouterSettings().blend_quality_weight,
        RouterSettings().blend_cost_weight,
        RouterSettings().blend_latency_weight,
        RouterSettings().request_cost_ceiling_usd,
        RouterSettings().canary_stable_model,
        RouterSettings().canary_model,
        RouterSettings().canary_weight,
        RouterSettings().latency_sla_ms,
    )

    strategy = strategies[RoutingStrategyName.REQUEST_CLASS_QOS]
    assert isinstance(strategy, RequestClassQosStrategy)
    assert strategy.strategy_name is RoutingStrategyName.REQUEST_CLASS_QOS
