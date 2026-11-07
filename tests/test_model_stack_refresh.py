"""Regression tests for the November 2026 provider model refresh.

Pins the behaviour nexus needs for the current OpenAI GPT-6, Anthropic Claude 5.5,
Google Gemini 4 / 3.8 and Kimi K3.x line-ups: tier classification, provider
parameter compatibility, list prices, and catalog/adapter consistency.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.anthropic import AnthropicAdapter
from adapters.base import BaseProviderAdapter
from adapters.google import GoogleGeminiAdapter
from adapters.moonshot import MoonshotAdapter
from adapters.openai import OpenAIAdapter
from router.config import default_model_catalog
from router.schemas import ChatMessage, RouterRequest
from router.strategies import (
    ModelTier,
    MultimodalInputPreferStrategy,
    StreamingPreferStrategy,
    StructuredOutputPreferStrategy,
    ToolCallingPreferStrategy,
    infer_model_tier,
)
from safety.circuit_breaker import CircuitBreakerRegistry
from safety.model_deprecation import ModelDeprecationGuard

CURRENT_MODELS = {
    "openai": ("gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna"),
    "anthropic": ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
    "google": ("gemini-4-argon", "gemini-3.8-flash"),
    "moonshot": ("kimi-k3.1-pro", "kimi-k3.1-pro-highspeed", "kimi-k3"),
}
# USD per 1M tokens (input, output) from the provider pricing pages on 2026-11-06.
LIST_PRICES: dict[str, tuple[float, float]] = {
    "gpt-6-astra": (10.0, 50.0),
    "gpt-6.1-sol": (2.0, 10.0),
    "gpt-6.1-terra": (1.0, 5.0),
    "gpt-6-luna": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "gemini-4-argon": (2.0, 10.0),
    "gemini-3.8-flash": (0.75, 3.75),
    "kimi-k3.1-pro": (2.0, 10.0),
    "kimi-k3.1-pro-highspeed": (4.0, 20.0),
    "kimi-k3": (3.0, 15.0),
}


def _adapters() -> dict[str, BaseProviderAdapter]:
    return {
        "openai": OpenAIAdapter(api_key="test-key", timeout_seconds=5.0),
        "anthropic": AnthropicAdapter(api_key="test-key", timeout_seconds=5.0),
        "google": GoogleGeminiAdapter(api_key="test-key", timeout_seconds=5.0),
        "moonshot": MoonshotAdapter(
            api_key="test-key", base_url="https://api.moonshot.ai/v1", timeout_seconds=5.0
        ),
    }


async def _captured_payload(
    module: str, adapter: BaseProviderAdapter, model: str
) -> dict[str, object]:
    captured: dict[str, dict[str, object]] = {}

    async def fake_post(*args: object, **kwargs: object) -> MagicMock:
        del args
        payload = kwargs.get("json")
        assert isinstance(payload, dict)
        captured["json"] = payload
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return response

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = fake_post
    with patch(f"adapters.{module}.httpx.AsyncClient", return_value=client):
        await adapter.complete(model, [ChatMessage(role="user", content="Hi.")], max_tokens=64)
    return captured["json"]


@pytest.mark.parametrize(
    ("model", "tier"),
    [
        ("gpt-6-astra", ModelTier.FRONTIER),
        ("gpt-6.1-sol", ModelTier.FRONTIER),
        ("gpt-6.1-terra", ModelTier.MID),
        ("gpt-6-luna", ModelTier.ECONOMY),
        ("claude-fable-5-1", ModelTier.FRONTIER),
        ("claude-opus-5-5", ModelTier.FRONTIER),
        ("claude-sonnet-5-5", ModelTier.FRONTIER),
        ("claude-haiku-5-5", ModelTier.MID),
        ("gemini-4-argon", ModelTier.FRONTIER),
        ("gemini-3.8-flash", ModelTier.ECONOMY),
        ("kimi-k3.1-pro", ModelTier.FRONTIER),
        ("kimi-k3.1-pro-highspeed", ModelTier.FRONTIER),
    ],
)
def test_infer_model_tier_classifies_current_lineups(model: str, tier: ModelTier) -> None:
    """Tier-based strategies must not demote current flagships to the mid tier."""
    assert infer_model_tier(model) is tier


@pytest.mark.parametrize("model", CURRENT_MODELS["openai"])
async def test_openai_adapter_sends_max_completion_tokens_for_gpt6(model: str) -> None:
    """GPT-6 chat completions take ``max_completion_tokens``, never legacy ``max_tokens``."""
    payload = await _captured_payload("openai", _adapters()["openai"], model)

    assert payload["max_completion_tokens"] == 64
    assert "max_tokens" not in payload


@pytest.mark.parametrize("model", CURRENT_MODELS["moonshot"])
async def test_moonshot_k3_payload_uses_max_completion_tokens_without_fixed_sampling(
    model: str,
) -> None:
    """Kimi K3.x: ``max_tokens`` is deprecated and fixed sampling params must be omitted."""
    payload = await _captured_payload("moonshot", _adapters()["moonshot"], model)

    assert payload["max_completion_tokens"] == 64
    assert "max_tokens" not in payload
    fixed = {"temperature", "top_p", "n", "presence_penalty", "frequency_penalty"}
    assert fixed.isdisjoint(payload)


@pytest.mark.parametrize(
    ("provider", "model"),
    [(provider, model) for provider, models in CURRENT_MODELS.items() for model in models],
)
def test_adapter_price_tables_match_provider_list_prices(provider: str, model: str) -> None:
    """Spend and budgets need real prices for every current model (unknown models cost $0)."""
    input_price, output_price = LIST_PRICES[model]

    cost = _adapters()[provider].estimate_cost(model, 1_000_000, 1_000_000)

    assert cost == pytest.approx(input_price + output_price)


def test_default_catalog_routes_to_current_models_with_matching_prices() -> None:
    """Catalog cost priors must equal the adapter price tables for every default model."""
    adapters = _adapters()
    catalog = default_model_catalog()

    assert set(catalog) == {
        "gpt-6-astra",
        "gpt-6-luna",
        "claude-opus-5-5",
        "claude-haiku-5-5",
        "gemini-4-argon",
        "gemini-3.8-flash",
        "kimi-k3.1-pro",
    }
    for model, candidate in catalog.items():
        expected = adapters[candidate.provider].estimate_cost(model, 1_000, 1_000)
        assert candidate.estimate_cost(1_000, 1_000) == pytest.approx(expected), model


@pytest.mark.parametrize(
    ("strategy_type", "method"),
    [
        (StructuredOutputPreferStrategy, "_supports_structured"),
        (ToolCallingPreferStrategy, "_supports_tools"),
        (StreamingPreferStrategy, "_supports_streaming"),
        (MultimodalInputPreferStrategy, "_supports_multimodal"),
    ],
)
def test_name_fallback_recognizes_gpt6_and_kimi_k3_models(
    strategy_type: type[StructuredOutputPreferStrategy]
    | type[ToolCallingPreferStrategy]
    | type[StreamingPreferStrategy]
    | type[MultimodalInputPreferStrategy],
    method: str,
) -> None:
    """Models outside the capability map still match the name-based fallbacks."""
    strategy = strategy_type(default_model_catalog(), CircuitBreakerRegistry(), capability_map={})
    request = RouterRequest(request_id="refresh", messages=[ChatMessage(content="hi")])

    for model in ("gpt-6.1-sol", "kimi-k3.1-pro-highspeed"):
        assert getattr(strategy, method)(model, request), model


@pytest.mark.parametrize(
    ("model", "replacement"),
    [
        ("kimi-k2", "kimi-k3.1-pro"),
        ("kimi-k2.6", "kimi-k3.1-pro"),
        ("kimi-k2.7-code", "kimi-k3.1-pro"),
        ("kimi-k2.7-code-highspeed", "kimi-k3.1-pro-highspeed"),
        ("moonshot-v1-128k", "kimi-k3.1-pro"),
        ("gpt-4o", "gpt-6.1-sol"),
        ("gpt-4o-mini", "gpt-6-luna"),
        ("claude-3-5-sonnet", "claude-sonnet-5-5"),
        ("gemini-1.5-pro", "gemini-4-argon"),
    ],
)
def test_deprecation_guard_points_retired_models_at_current_replacements(
    model: str, replacement: str
) -> None:
    """Kimi K2 is closed to new users and retires by end of 2026; replacements are current."""
    verdict = ModelDeprecationGuard().check(model)

    assert verdict.is_deprecated is True
    assert verdict.replacement == replacement
    assert verdict.severity == "warn"
