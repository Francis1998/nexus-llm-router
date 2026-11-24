"""Deprecated model ID → replacement guard (warn, or block from a retirement date)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

DeprecationSeverity = Literal["warn", "block"]


@dataclass(frozen=True, slots=True)
class DeprecationVerdict:
    """Outcome of checking a model ID against the deprecation map.

    ``retires_on`` is the provider's retirement date when the entry has one.
    From that date on (UTC) ``severity`` is ``"block"``, because requests to a
    retired model fail at the provider.
    """

    model_id: str
    is_deprecated: bool
    replacement: str | None
    severity: DeprecationSeverity | None
    reason: str | None
    retires_on: date | None = None


@dataclass(frozen=True, slots=True)
class _DeprecationEntry:
    """Internal curated / registered deprecation mapping."""

    replacement: str
    severity: DeprecationSeverity
    reason: str
    retires_on: date | None = None


def _utc_today() -> date:
    """Return the current calendar date in UTC."""
    return datetime.now(UTC).date()


def _claude_entry(
    model: str, *, deprecated_on: date, retires_on: date, replacement: str
) -> _DeprecationEntry:
    """Return an Anthropic deprecation that warns until it retires, then blocks."""
    return _DeprecationEntry(
        replacement=replacement,
        severity="warn",
        reason=(
            f"Anthropic deprecated {model} on {deprecated_on.isoformat()}; its Claude API "
            f"retirement date is {retires_on.isoformat()}, and requests to a retired model "
            f"fail; migrate to {replacement}"
        ),
        retires_on=retires_on,
    )


# Small curated seed of legacy aliases → current SKUs (November 2026 line-ups,
# lifecycle verified 2026-11-24). Distinct from ``router.model_ids`` catalog
# sync — this is a safety guard for callers still sending retired IDs, not a
# capability catalog refresh. Superseded models that their provider still serves
# (gpt-6-astra → gpt-6.2-astra, gpt-6.1-sol → gpt-6.2-sol, claude-fable-5-1 →
# claude-fable-5-2) are not deprecated, so they are deliberately absent.
#
# Entries with a ``retires_on`` date warn until that date and block from it.
# Claude dates are the Claude API's (Anthropic-operated platforms); partner
# platforms such as Amazon Bedrock set their own schedules. Source:
# https://platform.claude.com/docs/en/about-claude/model-deprecations
_CLAUDE_SONNET_3_5 = _claude_entry(
    "Claude Sonnet 3.5",
    deprecated_on=date(2025, 8, 13),
    retires_on=date(2025, 10, 28),
    replacement="claude-sonnet-5-5",
)
_CLAUDE_SONNET_4_5 = _claude_entry(
    "Claude Sonnet 4.5",
    deprecated_on=date(2026, 9, 30),
    retires_on=date(2026, 11, 30),
    replacement="claude-sonnet-5-5",
)
_CLAUDE_OPUS_4_5 = _claude_entry(
    "Claude Opus 4.5",
    deprecated_on=date(2026, 11, 23),
    retires_on=date(2027, 1, 22),
    replacement="claude-opus-5-5",
)
_KIMI_K2_RETIREMENT = (
    "Kimi K2 series is closed to new users since 2026-10-31 and retires by the end of December 2026"
)
_DEFAULT_DEPRECATIONS: dict[str, _DeprecationEntry] = {
    "gpt-4o": _DeprecationEntry(
        replacement="gpt-6.2-sol",
        severity="warn",
        reason="gpt-4o is deprecated; prefer gpt-6.2-sol",
    ),
    "gpt-4o-mini": _DeprecationEntry(
        replacement="gpt-6-luna",
        severity="warn",
        reason="gpt-4o-mini is deprecated; prefer gpt-6-luna",
    ),
    "claude-3-5-sonnet": _CLAUDE_SONNET_3_5,
    "claude-3-5-sonnet-20241022": _CLAUDE_SONNET_3_5,
    "claude-sonnet-4-5": _CLAUDE_SONNET_4_5,
    "claude-sonnet-4-5-20250929": _CLAUDE_SONNET_4_5,
    "claude-opus-4-5": _CLAUDE_OPUS_4_5,
    "claude-opus-4-5-20251101": _CLAUDE_OPUS_4_5,
    "gemini-1.5-pro": _DeprecationEntry(
        replacement="gemini-4-argon",
        severity="warn",
        reason="gemini-1.5-pro is deprecated; prefer gemini-4-argon",
    ),
    "gemini-2.5-pro": _DeprecationEntry(
        replacement="gemini-4-argon",
        severity="warn",
        reason=(
            "gemini-2.5-pro is deprecated and from 2026-12-01 is served by gemini-3.8-flash "
            "at 3.8 Flash rates; prefer gemini-4-argon"
        ),
    ),
    "moonshot-v1-128k": _DeprecationEntry(
        replacement="kimi-k3.1-pro",
        severity="warn",
        reason="moonshot-v1-128k is deprecated; prefer kimi-k3.1-pro",
    ),
    "kimi-k2": _DeprecationEntry(
        replacement="kimi-k3.1-pro",
        severity="warn",
        reason=f"{_KIMI_K2_RETIREMENT}; prefer kimi-k3.1-pro",
    ),
    "kimi-k2.6": _DeprecationEntry(
        replacement="kimi-k3.1-pro",
        severity="warn",
        reason=f"{_KIMI_K2_RETIREMENT}; prefer kimi-k3.1-pro",
    ),
    "kimi-k2.7-code": _DeprecationEntry(
        replacement="kimi-k3.1-pro",
        severity="warn",
        reason=f"{_KIMI_K2_RETIREMENT}; prefer kimi-k3.1-pro",
    ),
    "kimi-k2.7-code-highspeed": _DeprecationEntry(
        replacement="kimi-k3.1-pro-highspeed",
        severity="warn",
        reason=f"{_KIMI_K2_RETIREMENT}; prefer kimi-k3.1-pro-highspeed",
    ),
}


class ModelDeprecationGuard:
    """Map deprecated model IDs to replacements with warn|block severity.

    Closes the LiteLLM / OpenRouter model-alias deprecation gap for offline
    Nexus gateways serving GPT-6 / Claude 5.5 / Gemini 4 / Kimi K3.1:
    callers ``check(model_id)`` before dispatch and either log a warning,
    rewrite to ``replacement``, or reject when ``severity == "block"``.
    Entries with a retirement date (``retires_on``) warn before that date and
    block on and after it, read from an injectable UTC-date clock.

    Distinct from ``router.model_ids`` / ``ProviderModelListSync`` catalog
    refresh — those keep known-good IDs in sync; this guard handles *retired*
    aliases that should not be routed as-is.
    """

    def __init__(
        self,
        *,
        seed: dict[str, tuple[str, DeprecationSeverity, str]] | None = None,
        include_defaults: bool = True,
        today: Callable[[], date] | None = None,
    ) -> None:
        """Initialize the guard with optional seed overrides.

        Args:
            seed: Optional ``model_id → (replacement, severity, reason)`` map
                merged on top of (or instead of) the built-in defaults. Seed
                entries have no retirement date; use ``register()`` for one.
            include_defaults: When ``True`` (default), start from the curated
                built-in deprecation map; when ``False``, start empty.
            today: Injectable clock returning the current date, compared with
                retirement dates in UTC (defaults to today's UTC date). An
                aware ``datetime`` is converted to UTC first; a naive one is
                taken as UTC.

        Raises:
            ValueError: If a seed entry has an empty id/replacement or invalid
                severity.
        """
        self._lock = threading.Lock()
        self._today = today or _utc_today
        self._entries: dict[str, _DeprecationEntry] = {}
        if include_defaults:
            self._entries.update(_DEFAULT_DEPRECATIONS)
        if seed is not None:
            for model_id, (replacement, severity, reason) in seed.items():
                self._validate(model_id, replacement, severity, reason)
                self._entries[model_id] = _DeprecationEntry(
                    replacement=replacement,
                    severity=severity,
                    reason=reason,
                )

    @staticmethod
    def _validate(
        model_id: str,
        replacement: str,
        severity: str,
        reason: str,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        if not replacement:
            raise ValueError("replacement must be non-empty")
        if severity not in ("warn", "block"):
            raise ValueError("severity must be 'warn' or 'block'")
        if not reason:
            raise ValueError("reason must be non-empty")

    @staticmethod
    def _validate_retires_on(retires_on: object) -> None:
        # ``datetime`` subclasses ``date`` but cannot be compared with one.
        if retires_on is not None and (
            isinstance(retires_on, datetime) or not isinstance(retires_on, date)
        ):
            raise ValueError("retires_on must be a datetime.date, not a datetime")

    def _current_date(self) -> date:
        current = self._today()
        if isinstance(current, datetime):
            if current.tzinfo is not None:
                current = current.astimezone(UTC)
            return current.date()
        return current

    def check(self, model_id: str) -> DeprecationVerdict:
        """Return a deprecation verdict for ``model_id``.

        Args:
            model_id: Requested model identifier (exact match, case-sensitive).

        Returns:
            ``DeprecationVerdict`` — ``is_deprecated`` is ``False`` when the ID
            is unknown to the guard (not an error). ``severity`` is ``"block"``
            on and after the entry's ``retires_on`` date.

        Raises:
            ValueError: If ``model_id`` is empty.
        """
        if not model_id:
            raise ValueError("model_id must be non-empty")
        with self._lock:
            entry = self._entries.get(model_id)
        if entry is None:
            return DeprecationVerdict(
                model_id=model_id,
                is_deprecated=False,
                replacement=None,
                severity=None,
                reason=None,
            )
        severity = entry.severity
        if entry.retires_on is not None and self._current_date() >= entry.retires_on:
            severity = "block"
        return DeprecationVerdict(
            model_id=model_id,
            is_deprecated=True,
            replacement=entry.replacement,
            severity=severity,
            reason=entry.reason,
            retires_on=entry.retires_on,
        )

    def register(
        self,
        model_id: str,
        replacement: str,
        *,
        severity: DeprecationSeverity = "warn",
        reason: str | None = None,
        retires_on: date | None = None,
    ) -> None:
        """Register or overwrite a deprecation mapping.

        Args:
            model_id: Deprecated model identifier.
            replacement: Preferred replacement model ID.
            severity: ``warn`` (advisory) or ``block`` (reject).
            reason: Human-readable explanation (auto-generated when omitted).
            retires_on: Optional provider retirement date; from that date on
                (UTC) the entry blocks even when ``severity`` is ``warn``.

        Raises:
            ValueError: If inputs are invalid.
        """
        self._validate_retires_on(retires_on)
        if reason:
            detail = reason
        elif retires_on is not None:
            detail = (
                f"{model_id} is deprecated (retirement date {retires_on.isoformat()}); "
                f"prefer {replacement}"
            )
        else:
            detail = f"{model_id} is deprecated; prefer {replacement}"
        self._validate(model_id, replacement, severity, detail)
        with self._lock:
            self._entries[model_id] = _DeprecationEntry(
                replacement=replacement,
                severity=severity,
                reason=detail,
                retires_on=retires_on,
            )

    def unregister(self, model_id: str) -> bool:
        """Remove a deprecation mapping.

        Args:
            model_id: Model identifier to remove.

        Returns:
            ``True`` if an entry was removed, ``False`` if unknown.

        Raises:
            ValueError: If ``model_id`` is empty.
        """
        if not model_id:
            raise ValueError("model_id must be non-empty")
        with self._lock:
            return self._entries.pop(model_id, None) is not None

    def known_deprecated(self) -> list[str]:
        """Return sorted deprecated model IDs currently registered."""
        with self._lock:
            return sorted(self._entries)
