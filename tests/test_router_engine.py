"""Tests for the Observe-Decide-Act router engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from router.config import RouterSettings
from router.engine import NexusRouter
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_BALANCED_MODEL
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
