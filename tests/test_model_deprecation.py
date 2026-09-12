"""Tests for ModelDeprecationGuard deprecated-ID → replacement mapping."""

from __future__ import annotations

import pytest

from safety.model_deprecation import DeprecationVerdict, ModelDeprecationGuard


def test_seed_aliases_warn_to_frontier_replacements() -> None:
    """Curated seeds map legacy aliases to current frontier SKUs with warn."""
    guard = ModelDeprecationGuard()
    gpt = guard.check("gpt-4o")
    assert isinstance(gpt, DeprecationVerdict)
    assert gpt.is_deprecated is True
    assert gpt.replacement == "gpt-5.5"
    assert gpt.severity == "warn"
    assert gpt.reason is not None

    claude = guard.check("claude-3-5-sonnet")
    assert claude.is_deprecated is True
    assert claude.replacement == "claude-sonnet-4.6"
    assert claude.severity == "warn"

    gemini = guard.check("gemini-1.5-pro")
    assert gemini.is_deprecated is True
    assert gemini.replacement == "gemini-3.1-pro-preview"

    kimi = guard.check("moonshot-v1-128k")
    assert kimi.is_deprecated is True
    assert kimi.replacement == "kimi-k2"


def test_current_frontier_ids_are_not_deprecated() -> None:
    """Current GPT-5.5 / Sonnet 4.6 / Gemini 3.x / Kimi K2 IDs pass clean."""
    guard = ModelDeprecationGuard()
    for model_id in (
        "gpt-5.5",
        "claude-sonnet-4.6",
        "gemini-3.1-pro-preview",
        "kimi-k2",
    ):
        verdict = guard.check(model_id)
        assert verdict.is_deprecated is False
        assert verdict.replacement is None
        assert verdict.severity is None
        assert verdict.reason is None


def test_register_and_unregister_extend_map() -> None:
    """register()/unregister() support operator-extensible deprecation maps."""
    guard = ModelDeprecationGuard(include_defaults=False)
    assert guard.known_deprecated() == []
    guard.register(
        "legacy-custom",
        "gpt-5.5",
        severity="block",
        reason="legacy-custom blocked; use gpt-5.5",
    )
    verdict = guard.check("legacy-custom")
    assert verdict.is_deprecated is True
    assert verdict.severity == "block"
    assert verdict.replacement == "gpt-5.5"
    assert guard.unregister("legacy-custom") is True
    assert guard.check("legacy-custom").is_deprecated is False
    assert guard.unregister("legacy-custom") is False


def test_seed_override_merges_on_defaults() -> None:
    """Constructor seed overrides / extends the curated defaults."""
    guard = ModelDeprecationGuard(
        seed={
            "gpt-4o": ("gpt-5.5", "block", "gpt-4o now blocked"),
            "old-flash": ("gemini-3.5-flash", "warn", "prefer gemini-3.5-flash"),
        }
    )
    blocked = guard.check("gpt-4o")
    assert blocked.severity == "block"
    assert blocked.reason == "gpt-4o now blocked"
    assert guard.check("old-flash").replacement == "gemini-3.5-flash"
    # Untouched default still present.
    assert guard.check("claude-3-5-sonnet").is_deprecated is True


def test_rejects_invalid_inputs() -> None:
    """Construction, check, register, and unregister validate inputs."""
    with pytest.raises(ValueError, match="severity"):
        ModelDeprecationGuard(
            include_defaults=False,
            seed={"x": ("gpt-5.5", "soft", "bad")},  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="model_id"):
        ModelDeprecationGuard(include_defaults=False).check("")
    guard = ModelDeprecationGuard(include_defaults=False)
    with pytest.raises(ValueError, match="replacement"):
        guard.register("old", "", severity="warn")
    with pytest.raises(ValueError, match="model_id"):
        guard.unregister("")
