"""Tests for the Observe-Decide-Act router engine."""

from pathlib import Path

import pytest

from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from router.config import RouterSettings
from router.engine import NexusRouter
from router.model_ids import ANTHROPIC_SAFETY_MODEL, OPENAI_BALANCED_MODEL
from router.schemas import ChatMessage, RouterRequest, RoutingStrategyName


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
