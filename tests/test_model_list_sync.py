"""Tests for offline provider model-list sync into the capability catalog."""

from __future__ import annotations

import pytest

from router.capability_catalog import _STATIC_SEED, ModelCapabilityCatalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
)
from router.model_list_sync import ModelDescriptor, ModelListSyncResult, ProviderModelListSync


def test_sync_static_loads_frontier_seed() -> None:
    """sync_static refreshes GPT-5.5 / Sonnet 4.6 / Gemini / Kimi into the catalog."""
    syncer = ProviderModelListSync()
    result = syncer.sync_static()
    assert isinstance(result, ModelListSyncResult)
    assert result.refreshed_from_static == len(_STATIC_SEED)
    assert result.upserted == 0
    assert OPENAI_FRONTIER_MODEL in result.model_ids
    assert ANTHROPIC_SAFETY_MODEL in result.model_ids
    assert GEMINI_FLASH_MODEL in result.model_ids
    assert MOONSHOT_BALANCED_MODEL in result.model_ids
    assert syncer.catalog.get(OPENAI_FRONTIER_MODEL) == _STATIC_SEED[OPENAI_FRONTIER_MODEL]


def test_sync_upserts_descriptors_without_network() -> None:
    """Provided descriptors are upserted deterministically with no network I/O."""
    catalog = ModelCapabilityCatalog(auto_seed=False)
    syncer = ProviderModelListSync(catalog)
    descriptors = [
        ModelDescriptor("gpt-5.5", frozenset({"tools", "vision"}), provider="openai"),
        ModelDescriptor("claude-sonnet-4-6", frozenset({"tools", "mcp"}), provider="anthropic"),
        ModelDescriptor("gemini-3.5-flash", frozenset({"vision", "audio"}), provider="google"),
        ModelDescriptor("kimi-k2", frozenset({"tools", "long_context"}), provider="moonshot"),
    ]
    result = syncer.sync(descriptors)
    assert result.refreshed_from_static == 0
    assert result.upserted == 4
    assert catalog.get("gpt-5.5") == frozenset({"tools", "vision"})
    assert catalog.get("claude-sonnet-4-6") == frozenset({"tools", "mcp"})
    assert catalog.get("gemini-3.5-flash") == frozenset({"vision", "audio"})
    assert catalog.get("kimi-k2") == frozenset({"tools", "long_context"})


def test_sync_refresh_static_first_then_upserts() -> None:
    """refresh_static_first restores the seed, then overlays provider descriptors."""
    catalog = ModelCapabilityCatalog(auto_seed=False)
    catalog.upsert("stale-model", {"tools"})
    syncer = ProviderModelListSync(catalog)
    result = syncer.sync(
        [ModelDescriptor("custom-router-model", frozenset({"tools", "json"}))],
        refresh_static_first=True,
    )
    assert result.refreshed_from_static == len(_STATIC_SEED)
    assert result.upserted == 1
    assert catalog.get("stale-model") is None
    assert catalog.get(OPENAI_FRONTIER_MODEL) == _STATIC_SEED[OPENAI_FRONTIER_MODEL]
    assert catalog.get("custom-router-model") == frozenset({"tools", "json"})


def test_sync_replaces_existing_capabilities() -> None:
    """Later descriptors win when the same model_id is upserted twice."""
    syncer = ProviderModelListSync()
    syncer.sync_static()
    syncer.sync(
        [
            ModelDescriptor(OPENAI_FRONTIER_MODEL, frozenset({"tools"})),
            ModelDescriptor(OPENAI_FRONTIER_MODEL, frozenset({"tools", "vision", "mcp"})),
        ]
    )
    assert syncer.catalog.get(OPENAI_FRONTIER_MODEL) == frozenset({"tools", "vision", "mcp"})


def test_model_descriptor_rejects_empty_model_id() -> None:
    """Empty model ids are rejected at descriptor construction."""
    with pytest.raises(ValueError, match="model_id"):
        ModelDescriptor("", frozenset({"tools"}))


def test_sync_rejects_non_descriptor_entries() -> None:
    """sync requires ModelDescriptor instances (not raw tuples/dicts)."""
    syncer = ProviderModelListSync()
    with pytest.raises(TypeError, match="ModelDescriptor"):
        syncer.sync([("gpt-5.5", {"tools"})])  # type: ignore[list-item]
