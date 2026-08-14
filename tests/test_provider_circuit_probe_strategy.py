"""Tests for the provider-circuit-probe routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_FRONTIER_MODEL
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
    ProviderCircuitProbeStrategy,
    build_strategies,
)


class _CircuitHealth:
    """Controllable closed/open/half-open provider health."""

    def __init__(self, states: dict[str, str] | None = None) -> None:
        self._states = states or {}

    def is_available(self, provider: str) -> bool:
        return self._states.get(provider, "closed") != "open"

    def is_half_open(self, provider: str) -> bool:
        return self._states.get(provider) == "half-open"


def _request() -> RouterRequest:
    return RouterRequest(request_id="req-provider-probe", messages=[ChatMessage(content="hello")])


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=100,
    )


def test_provider_circuit_probe_enum_parses() -> None:
    assert (
        RoutingStrategyName("provider-circuit-probe") is RoutingStrategyName.PROVIDER_CIRCUIT_PROBE
    )


def test_provider_circuit_probe_keeps_closed_quality_leader() -> None:
    strategy = ProviderCircuitProbeStrategy(default_model_catalog(), _CircuitHealth())

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "closed quality leader" in decision.rationale


def test_provider_circuit_probe_open_leader_selects_healthy_alternate() -> None:
    strategy = ProviderCircuitProbeStrategy(
        default_model_catalog(),
        _CircuitHealth({"anthropic": "open"}),
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "circuit open" in decision.rationale
    assert "actively selected healthy alternate" in decision.rationale


def test_provider_circuit_probe_allows_half_open_leader_within_budget() -> None:
    strategy = ProviderCircuitProbeStrategy(
        default_model_catalog(),
        _CircuitHealth({"anthropic": "half-open"}),
        probe_budget=1,
    )

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "probe 1/1" in decision.rationale


def test_provider_circuit_probe_exhausted_budget_falls_back() -> None:
    strategy = ProviderCircuitProbeStrategy(
        default_model_catalog(),
        _CircuitHealth({"anthropic": "half-open"}),
        probe_budget=1,
    )
    strategy.choose(_request(), _signals())

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert "exhausted half-open probe budget 1/1" in decision.rationale
    assert "healthy openai" in decision.rationale


def test_provider_circuit_probe_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        ProviderCircuitProbeStrategy(default_model_catalog(), _CircuitHealth(), probe_budget=0)


def test_provider_circuit_probe_registered_by_strategy_factory() -> None:
    settings = RouterSettings(provider_circuit_probe_budget=3)
    health = _CircuitHealth()
    strategies = build_strategies(
        default_model_catalog(),
        LatencyStats(),
        InflightStats(),
        settings.quality_floor,
        settings.ab_model_a,
        settings.ab_model_b,
        settings.ab_model_a_weight,
        health,
        settings.blend_quality_weight,
        settings.blend_cost_weight,
        settings.blend_latency_weight,
        settings.request_cost_ceiling_usd,
        settings.canary_stable_model,
        settings.canary_model,
        settings.canary_weight,
        settings.latency_sla_ms,
        provider_circuit_probe_budget=settings.provider_circuit_probe_budget,
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_CIRCUIT_PROBE]
    assert isinstance(strategy, ProviderCircuitProbeStrategy)
    assert strategy._probe_budget == 3  # noqa: SLF001
    assert RouterSettings().provider_circuit_probe_budget == 1
