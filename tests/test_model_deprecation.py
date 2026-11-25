"""Tests for ModelDeprecationGuard deprecated-ID → replacement mapping."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from safety.model_deprecation import DeprecationVerdict, ModelDeprecationGuard

# (model id, replacement, deprecation date, Claude API retirement date), from
# https://platform.claude.com/docs/en/about-claude/model-deprecations (2026-11-24).
CLAUDE_4_5_LIFECYCLE = [
    ("claude-sonnet-4-5", "claude-sonnet-5-5", "2026-09-30", date(2026, 11, 30)),
    ("claude-sonnet-4-5-20250929", "claude-sonnet-5-5", "2026-09-30", date(2026, 11, 30)),
    ("claude-opus-4-5", "claude-opus-5-5", "2026-11-23", date(2027, 1, 22)),
    ("claude-opus-4-5-20251101", "claude-opus-5-5", "2026-11-23", date(2027, 1, 22)),
]


def test_seed_aliases_map_to_frontier_replacements() -> None:
    """Curated seeds map legacy aliases to current frontier SKUs."""
    guard = ModelDeprecationGuard()
    gpt = guard.check("gpt-4o")
    assert isinstance(gpt, DeprecationVerdict)
    assert gpt.is_deprecated is True
    assert gpt.replacement == "gpt-6.2-sol"
    assert gpt.severity == "warn"
    assert gpt.reason is not None

    claude = guard.check("claude-3-5-sonnet")
    assert claude.is_deprecated is True
    assert claude.replacement == "claude-sonnet-5-5"
    # Anthropic retired Claude Sonnet 3.5 on 2025-10-28, so the guard blocks it.
    assert claude.severity == "block"

    gemini = guard.check("gemini-1.5-pro")
    assert gemini.is_deprecated is True
    assert gemini.replacement == "gemini-4-argon"

    kimi = guard.check("moonshot-v1-128k")
    assert kimi.is_deprecated is True
    assert kimi.replacement == "kimi-k3.1-pro"


def test_current_frontier_ids_are_not_deprecated() -> None:
    """Current GPT-6.2 / Claude 5.5 / Gemini 4 / Kimi K3.1 IDs and superseded GPT-6 IDs pass."""
    guard = ModelDeprecationGuard()
    for model_id in (
        "gpt-6.2-astra",
        "gpt-6.2-sol",
        "gpt-6-astra",
        "gpt-6.1-sol",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "gemini-4-argon",
        "gemini-4-neon",
        "gemini-3.8-flash",
        "kimi-k3.1-pro",
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


@pytest.mark.parametrize(
    ("model", "replacement", "deprecated_on", "retires_on"), CLAUDE_4_5_LIFECYCLE
)
def test_claude_4_5_ids_are_flagged_with_their_lifecycle_dates(
    model: str, replacement: str, deprecated_on: str, retires_on: date
) -> None:
    """Anthropic deprecated Claude Sonnet 4.5 and Opus 4.5; the guard must say so."""
    verdict = ModelDeprecationGuard().check(model)

    assert verdict.is_deprecated is True, f"{model} passes the guard silently"
    assert verdict.replacement == replacement
    assert verdict.retires_on == retires_on
    assert verdict.reason is not None
    for fact in (deprecated_on, retires_on.isoformat(), replacement):
        assert fact in verdict.reason


@pytest.mark.parametrize(
    ("model", "replacement", "retires_on"),
    [
        (model, replacement, retires_on)
        for model, replacement, _, retires_on in CLAUDE_4_5_LIFECYCLE
    ],
)
def test_claude_4_5_ids_warn_until_retirement_then_block(
    model: str, replacement: str, retires_on: date
) -> None:
    """Severity turns from warn to block on the retirement date (UTC), not before."""
    today = [retires_on - timedelta(days=1)]
    guard = ModelDeprecationGuard(today=lambda: today[0])

    assert guard.check(model).severity == "warn"
    today[0] = retires_on
    blocked = guard.check(model)
    assert blocked.severity == "block"
    assert blocked.replacement == replacement
    assert replacement in (blocked.reason or "")
    today[0] = retires_on + timedelta(days=365)
    assert guard.check(model).severity == "block"


@pytest.mark.parametrize("model", ["claude-3-5-sonnet", "claude-3-5-sonnet-20241022"])
def test_claude_sonnet_3_5_blocks_after_its_2025_retirement(model: str) -> None:
    """Anthropic retired Claude Sonnet 3.5 on 2025-10-28; requests to it fail."""
    verdict = ModelDeprecationGuard().check(model)

    assert verdict.severity == "block"
    assert verdict.retires_on == date(2025, 10, 28)
    assert "claude-sonnet-5-5" in (verdict.reason or "")


@pytest.mark.parametrize("model", ["gpt-4o", "gemini-2.5-pro", "moonshot-v1-128k", "kimi-k2"])
def test_entries_without_a_retirement_date_keep_their_severity(model: str) -> None:
    """Undated entries never escalate (Kimi K2 retires "by the end of December 2026")."""
    verdict = ModelDeprecationGuard(today=lambda: date(2030, 1, 1)).check(model)

    assert verdict.severity == "warn"
    assert verdict.retires_on is None


def test_registered_retirement_date_escalates_on_the_utc_date() -> None:
    """Operators can date their own entries; the clock is read as a UTC date."""
    pacific = timezone(timedelta(hours=-8))
    now = [datetime(2026, 12, 30, 15, 59, tzinfo=pacific)]
    guard = ModelDeprecationGuard(include_defaults=False, today=lambda: now[0])
    guard.register("legacy-x", "kimi-k3.1-pro", retires_on=date(2026, 12, 31))
    guard.register("legacy-y", "gpt-6.2-sol", severity="block", retires_on=date(2027, 6, 1))

    warned = guard.check("legacy-x")
    assert warned.severity == "warn"
    assert warned.retires_on == date(2026, 12, 31)
    assert warned.reason is not None
    assert "2026-12-31" in warned.reason
    assert "kimi-k3.1-pro" in warned.reason
    # An explicit block applies before the retirement date too.
    assert guard.check("legacy-y").severity == "block"
    # 16:00 in UTC-8 on 2026-12-30 is already 2026-12-31 in UTC.
    now[0] = datetime(2026, 12, 30, 16, 0, tzinfo=pacific)
    assert guard.check("legacy-x").severity == "block"
    now[0] = datetime(2026, 12, 31, 0, 0, tzinfo=UTC)
    assert guard.check("legacy-x").severity == "block"

    with pytest.raises(ValueError, match="retires_on"):
        guard.register("legacy-z", "gpt-6.2-sol", retires_on=datetime(2027, 1, 1, tzinfo=UTC))
