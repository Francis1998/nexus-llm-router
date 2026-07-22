"""Tests for the Observe-Decide-Act router engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from router.config import RouterSettings
from router.engine import NexusRouter, RoutingFailedError
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    OPENAI_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
)
from router.schemas import ChatMessage, RouterRequest, RoutingStrategyName
from router.state import RequestState

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


class FakeMetricIncrementer:
    """Metric child that records no-op increments."""

    def inc(self) -> None:
        """Increment the fake metric."""


class FakeRouterRequestsMetric:
    """Test double for the router request counter."""

    def __init__(self) -> None:
        """Initialize captured label values."""

        self.labels_seen: list[tuple[str, str]] = []

    def labels(self, strategy: str, state: str) -> FakeMetricIncrementer:
        """Record labels and return an incrementable metric child."""

        self.labels_seen.append((strategy, state))
        return FakeMetricIncrementer()


class FakeProviderErrorRate:
    """Test double for the provider error-rate counter."""

    def __init__(self) -> None:
        """Initialize captured label values."""

        self.labels_seen: list[tuple[str, str]] = []

    def labels(self, provider: str, model: str) -> FakeMetricIncrementer:
        """Record provider/model labels and return an incrementable child."""

        self.labels_seen.append((provider, model))
        return FakeMetricIncrementer()


@pytest.mark.asyncio
async def test_rule_based_routes_medical_prompt_to_claude(tmp_path: Path) -> None:
    """Rule-based routing should prioritize Claude for medical prompts."""
    router = NexusRouter(
        settings=RouterSettings(audit_log_path=str(tmp_path / "audit.jsonl")),
        adapter_registry=AdapterRegistry(
            {
                "openai": MockProviderAdapter("openai"),
                "anthropic": MockProviderAdapter("anthropic"),
                "google": MockProviderAdapter("google"),
                "moonshot": MockProviderAdapter("moonshot"),
            },
        ),
    )
    response = await router.complete(
        RouterRequest(
            request_id="req-medical",
            messages=[ChatMessage(content="Assess patient symptoms and treatment risk.")],
            strategy=RoutingStrategyName.RULE_BASED,
        ),
    )
    assert response.model_used == ANTHROPIC_SAFETY_MODEL
    assert response.routing_strategy == RoutingStrategyName.RULE_BASED
    assert "medical" in response.rationale
    assert (tmp_path / "audit.jsonl").exists()


@pytest.mark.asyncio
async def test_fallback_chain_uses_next_provider_on_failure(tmp_path: Path) -> None:
    """Router should fallback when the selected provider fails."""
    router = NexusRouter(
        settings=RouterSettings(audit_log_path=str(tmp_path / "audit.jsonl")),
        adapter_registry=AdapterRegistry(
            {
                "openai": MockProviderAdapter("openai", fail=True),
                "anthropic": MockProviderAdapter("anthropic"),
                "google": MockProviderAdapter("google"),
                "moonshot": MockProviderAdapter("moonshot"),
            },
        ),
    )
    response = await router.complete(
        RouterRequest(
            request_id="req-code",
            messages=[ChatMessage(content="Debug this Python class with async retries and tests.")],
            strategy=RoutingStrategyName.RULE_BASED,
        ),
    )
    assert response.model_used == ANTHROPIC_SAFETY_MODEL
    assert "fallback attempt" in response.rationale


@pytest.mark.asyncio
async def test_fallback_chain_uses_next_provider_when_selected_adapter_missing(
    tmp_path: Path,
) -> None:
    """Router should fallback when the selected provider is not registered."""
    router = NexusRouter(
        settings=RouterSettings(audit_log_path=str(tmp_path / "audit.jsonl")),
        adapter_registry=AdapterRegistry(
            {
                "anthropic": MockProviderAdapter("anthropic"),
                "google": MockProviderAdapter("google"),
                "moonshot": MockProviderAdapter("moonshot"),
            },
        ),
    )

    response = await router.complete(
        RouterRequest(
            request_id="req-missing-openai",
            messages=[ChatMessage(content="Debug this Python class with async retries and tests.")],
            strategy=RoutingStrategyName.RULE_BASED,
        ),
    )

    assert response.model_used == ANTHROPIC_SAFETY_MODEL
    assert "fallback attempt" in response.rationale


@pytest.mark.asyncio
async def test_cost_optimal_respects_quality_floor(tmp_path: Path) -> None:
    """Cost-optimal routing should choose the cheapest model above quality floor."""
    router = NexusRouter(
        settings=RouterSettings(
            audit_log_path=str(tmp_path / "audit.jsonl"),
            quality_floor=0.8,
        ),
        adapter_registry=AdapterRegistry(
            {
                "openai": MockProviderAdapter("openai"),
                "anthropic": MockProviderAdapter("anthropic"),
                "google": MockProviderAdapter("google"),
                "moonshot": MockProviderAdapter("moonshot"),
            },
        ),
    )
    response = await router.complete(
        RouterRequest(
            request_id="req-cost",
            messages=[ChatMessage(content="Summarize the following operational update.")],
            strategy=RoutingStrategyName.COST_OPTIMAL,
        ),
    )
    assert response.model_used == OPENAI_BALANCED_MODEL
    assert response.routing_strategy == RoutingStrategyName.COST_OPTIMAL


@pytest.mark.asyncio
async def test_request_metrics_use_selected_strategy_label(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Router request metrics should not emit an unknown strategy label."""

    metric = FakeRouterRequestsMetric()
    monkeypatch.setattr("router.engine.router_requests_total", metric)
    router = NexusRouter(
        settings=RouterSettings(audit_log_path=str(tmp_path / "audit.jsonl")),
        adapter_registry=AdapterRegistry(
            {
                "openai": MockProviderAdapter("openai"),
                "anthropic": MockProviderAdapter("anthropic"),
                "google": MockProviderAdapter("google"),
                "moonshot": MockProviderAdapter("moonshot"),
            },
        ),
    )

    await router.complete(
        RouterRequest(
            request_id="req-metrics",
            messages=[ChatMessage(content="Summarize this infrastructure incident.")],
            strategy=RoutingStrategyName.RULE_BASED,
        ),
    )

    assert metric.labels_seen == [
        (RoutingStrategyName.RULE_BASED.value, RequestState.RECEIVED.value),
        (RoutingStrategyName.RULE_BASED.value, RequestState.RESPONDED.value),
    ]


@pytest.mark.asyncio
async def test_budget_rejection_is_not_counted_as_provider_error(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A budget-cap rejection must not be recorded as a provider fault.

    A budget cap is a client-side guardrail, not a provider outage. Recording it
    as a provider error (and tripping the circuit breaker) would let repeated
    budget rejections open a healthy provider's circuit and degrade unrelated
    traffic. With the cap set to zero every candidate is over budget, so the
    request fails without any provider-error metric or circuit-breaker failure.
    """

    error_rate = FakeProviderErrorRate()
    monkeypatch.setattr("router.engine.provider_error_rate", error_rate)
    router = NexusRouter(
        settings=RouterSettings(
            audit_log_path=str(tmp_path / "audit.jsonl"),
            budget_cap_usd=0.0,
        ),
        adapter_registry=AdapterRegistry(
            {
                "openai": MockProviderAdapter("openai"),
                "anthropic": MockProviderAdapter("anthropic"),
                "google": MockProviderAdapter("google"),
                "moonshot": MockProviderAdapter("moonshot"),
            },
        ),
    )

    with pytest.raises(RoutingFailedError):
        await router.complete(
            RouterRequest(
                request_id="req-over-budget",
                messages=[ChatMessage(content="Summarize this infrastructure incident.")],
                strategy=RoutingStrategyName.RULE_BASED,
            ),
        )

    assert error_rate.labels_seen == []
    assert all(
        state.consecutive_failures == 0 and state.opened_at is None
        for state in router._circuit_breakers._states.values()
    )


@pytest.mark.asyncio
async def test_inflight_counter_is_cleared_after_provider_failure(tmp_path: Path) -> None:
    """A failed provider attempt must not leak least-busy load.

    The least-busy strategy reads live in-flight counters. If a provider failure
    left its counter incremented, future routing would incorrectly avoid that
    provider after the request had already finished.
    """
    router = NexusRouter(
        settings=RouterSettings(audit_log_path=str(tmp_path / "audit.jsonl")),
        adapter_registry=AdapterRegistry(
            {
                "openai": MockProviderAdapter("openai"),
                "anthropic": MockProviderAdapter("anthropic", fail=True),
                "google": MockProviderAdapter("google"),
                "moonshot": MockProviderAdapter("moonshot"),
            },
        ),
    )

    response = await router.complete(
        RouterRequest(
            request_id="req-least-busy-fallback",
            messages=[ChatMessage(content="Summarize this infrastructure incident.")],
            strategy=RoutingStrategyName.LEAST_BUSY,
        ),
    )

    assert response.model_used == OPENAI_FRONTIER_MODEL
    assert all(
        router._inflight_stats.load_score(provider) == 0
        for provider in {"openai", "anthropic", "google", "moonshot"}
    )
