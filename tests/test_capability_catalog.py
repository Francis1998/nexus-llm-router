"""Tests for the refreshable model capability catalog."""

from __future__ import annotations

import threading

from router.capability_catalog import _STATIC_SEED, ModelCapabilityCatalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
)


def test_auto_seed_includes_known_frontier_models() -> None:
    """Default catalog seeds GPT-5.5 / Claude Sonnet 4.6 / Gemini / Kimi."""
    catalog = ModelCapabilityCatalog()
    assert OPENAI_FRONTIER_MODEL in catalog
    assert ANTHROPIC_SAFETY_MODEL in catalog
    assert GEMINI_FLASH_MODEL in catalog
    assert MOONSHOT_BALANCED_MODEL in catalog
    assert len(catalog) == len(_STATIC_SEED)
    gpt = catalog.get(OPENAI_FRONTIER_MODEL)
    assert gpt is not None
    assert "tools" in gpt
    assert "vision" in gpt


def test_get_unknown_returns_none() -> None:
    """Unknown models return None."""
    catalog = ModelCapabilityCatalog()
    assert catalog.get("not-a-real-model") is None


def test_upsert_adds_and_replaces_capabilities() -> None:
    """upsert inserts new models and replaces existing frozensets."""
    catalog = ModelCapabilityCatalog(auto_seed=False)
    assert len(catalog) == 0
    catalog.upsert("gpt-5.5", {"tools", "json"})
    assert catalog.get("gpt-5.5") == frozenset({"tools", "json"})
    catalog.upsert("gpt-5.5", {"tools", "vision", "streaming"})
    assert catalog.get("gpt-5.5") == frozenset({"tools", "vision", "streaming"})


def test_refresh_from_static_restores_seed() -> None:
    """refresh_from_static replaces custom entries with the static seed."""
    catalog = ModelCapabilityCatalog(auto_seed=False)
    catalog.upsert("custom-model", {"tools"})
    catalog.upsert(OPENAI_FRONTIER_MODEL, {"only_tools"})
    loaded = catalog.refresh_from_static()
    assert loaded == len(_STATIC_SEED)
    assert catalog.get("custom-model") is None
    assert catalog.get(OPENAI_FRONTIER_MODEL) == _STATIC_SEED[OPENAI_FRONTIER_MODEL]


def test_snapshot_is_isolated_from_later_mutations() -> None:
    """snapshot returns a copy that strategies can hold safely."""
    catalog = ModelCapabilityCatalog()
    snap = catalog.snapshot()
    assert snap[OPENAI_FRONTIER_MODEL] == catalog.get(OPENAI_FRONTIER_MODEL)
    catalog.upsert(OPENAI_FRONTIER_MODEL, {"tools"})
    assert "vision" in snap[OPENAI_FRONTIER_MODEL]
    assert catalog.get(OPENAI_FRONTIER_MODEL) == frozenset({"tools"})


def test_thread_safe_upsert_and_snapshot() -> None:
    """Concurrent upserts and snapshots do not raise or corrupt state."""
    catalog = ModelCapabilityCatalog(auto_seed=False)
    errors: list[BaseException] = []

    def writer(start: int) -> None:
        try:
            for index in range(start, start + 50):
                catalog.upsert(f"model-{index}", {f"cap-{index}"})
                _ = catalog.snapshot()
        except BaseException as exc:  # pragma: no cover - surfaced via errors
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i * 50,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(catalog) == 200
