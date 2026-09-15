"""Tests for ModelCapabilityMatcher fit / partial / mismatch verdicts."""

from __future__ import annotations

import pytest

from safety.capability_match import CapabilityMatchAdvice, ModelCapabilityMatcher


def test_fit_partial_and_mismatch_verdicts() -> None:
    """Request needs map to fit / partial / mismatch against model flags."""
    matcher = ModelCapabilityMatcher(
        catalog={
            "gpt-5.5": {"vision", "tools", "json_schema", "long_context"},
            "claude-sonnet-4-6": {"tools", "json_schema", "long_context"},
            "gemini-3.5-flash": {"vision", "long_context"},
            "kimi-k2": {"tools"},
        }
    )
    fit = matcher.match(model="gpt-5.5", needs=["vision", "tools"])
    assert isinstance(fit, CapabilityMatchAdvice)
    assert fit.verdict == "fit"
    assert fit.missing == frozenset()

    partial = matcher.match(
        model="claude-sonnet-4-6",
        needs=["vision", "tools", "json_schema"],
    )
    assert partial.verdict == "partial"
    assert partial.missing == frozenset({"vision"})
    assert partial.matched == frozenset({"tools", "json_schema"})

    mismatch = matcher.match(model="kimi-k2", needs=["vision", "json_schema"])
    assert mismatch.verdict == "mismatch"
    assert mismatch.matched == frozenset()
    assert "mismatch" in mismatch.advisory


def test_per_call_capability_override_and_empty_needs() -> None:
    """Per-call capabilities override the catalog; empty needs are a fit."""
    matcher = ModelCapabilityMatcher(catalog={"gemini-3.x": {"vision"}})
    override = matcher.match(
        model="gemini-3.x",
        needs=["tools"],
        capabilities=["tools", "json_schema"],
    )
    assert override.verdict == "fit"
    assert override.supported == frozenset({"tools", "json_schema"})
    empty = matcher.match(model="gemini-3.x", needs=[])
    assert empty.verdict == "fit"
    assert matcher.capabilities_for("gemini-3.x") == frozenset({"vision"})
    assert matcher.capabilities_for("missing-model") == frozenset()


def test_catalog_models_include_frontier_ids() -> None:
    """Catalog retains GPT-5.5 / Sonnet 4.6 / Gemini 3.x / Kimi K2 ids."""
    matcher = ModelCapabilityMatcher(
        catalog={
            "gpt-5.5": ["vision", "tools"],
            "claude-sonnet-4-6": ["tools", "json_schema"],
            "gemini-3.x": ["vision", "long_context"],
            "kimi-k2": ["long_context"],
        }
    )
    assert matcher.models() == [
        "gpt-5.5",
        "claude-sonnet-4-6",
        "gemini-3.x",
        "kimi-k2",
    ]
    long_ctx = matcher.match(model="kimi-k2", needs=["long_context"])
    assert long_ctx.verdict == "fit"


def test_rejects_unknown_and_invalid_inputs() -> None:
    """Construction and match validate catalog flags and request needs."""
    with pytest.raises(ValueError, match="known_capabilities"):
        ModelCapabilityMatcher(known_capabilities=[])
    with pytest.raises(ValueError, match="catalog model"):
        ModelCapabilityMatcher(catalog={"": ["tools"]})
    with pytest.raises(ValueError, match="unknown capability"):
        ModelCapabilityMatcher(catalog={"gpt-5.5": ["audio"]})
    matcher = ModelCapabilityMatcher()
    with pytest.raises(ValueError, match="model"):
        matcher.match(model="", needs=["tools"])
    with pytest.raises(ValueError, match="unknown needs"):
        matcher.match(model="gpt-5.5", needs=["audio"])
    with pytest.raises(ValueError, match="unknown capabilities"):
        matcher.match(model="gpt-5.5", needs=["tools"], capabilities=["audio"])
    with pytest.raises(ValueError, match="needs entries"):
        matcher.match(model="gpt-5.5", needs=[""])


def test_models_list_and_capabilities_for() -> None:
    """models() preserves insertion order; capabilities_for validates model id."""
    matcher = ModelCapabilityMatcher(
        catalog={
            "gpt-5.5": ["vision"],
            "claude-sonnet-4-6": ["tools"],
        }
    )
    assert matcher.models() == ["gpt-5.5", "claude-sonnet-4-6"]
    assert matcher.capabilities_for("gpt-5.5") == frozenset({"vision"})
    with pytest.raises(ValueError, match="model"):
        matcher.capabilities_for("")
