"""Deprecated model ID → replacement guard (warn or block)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

DeprecationSeverity = Literal["warn", "block"]


@dataclass(frozen=True, slots=True)
class DeprecationVerdict:
    """Outcome of checking a model ID against the deprecation map."""

    model_id: str
    is_deprecated: bool
    replacement: str | None
    severity: DeprecationSeverity | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class _DeprecationEntry:
    """Internal curated / registered deprecation mapping."""

    replacement: str
    severity: DeprecationSeverity
    reason: str


# Small curated seed of legacy aliases → current frontier SKUs.
# Distinct from ``router.model_ids`` catalog sync — this is a safety guard for
# callers still sending retired IDs, not a capability catalog refresh.
_DEFAULT_DEPRECATIONS: dict[str, _DeprecationEntry] = {
    "gpt-4o": _DeprecationEntry(
        replacement="gpt-5.5",
        severity="warn",
        reason="gpt-4o is deprecated; prefer gpt-5.5",
    ),
    "gpt-4o-mini": _DeprecationEntry(
        replacement="gpt-5.5",
        severity="warn",
        reason="gpt-4o-mini is deprecated; prefer gpt-5.5",
    ),
    "claude-3-5-sonnet": _DeprecationEntry(
        replacement="claude-sonnet-4.6",
        severity="warn",
        reason="claude-3-5-sonnet is deprecated; prefer claude-sonnet-4.6",
    ),
    "claude-3-5-sonnet-20241022": _DeprecationEntry(
        replacement="claude-sonnet-4.6",
        severity="warn",
        reason="claude-3-5-sonnet dated alias is deprecated; prefer claude-sonnet-4.6",
    ),
    "gemini-1.5-pro": _DeprecationEntry(
        replacement="gemini-3.1-pro-preview",
        severity="warn",
        reason="gemini-1.5-pro is deprecated; prefer gemini-3.x",
    ),
    "moonshot-v1-128k": _DeprecationEntry(
        replacement="kimi-k2",
        severity="warn",
        reason="moonshot-v1-128k is deprecated; prefer kimi-k2",
    ),
}


class ModelDeprecationGuard:
    """Map deprecated model IDs to replacements with warn|block severity.

    Closes the LiteLLM / OpenRouter model-alias deprecation gap for offline
    Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2:
    callers ``check(model_id)`` before dispatch and either log a warning,
    rewrite to ``replacement``, or reject when ``severity == "block"``.

    Distinct from ``router.model_ids`` / ``ProviderModelListSync`` catalog
    refresh — those keep known-good IDs in sync; this guard handles *retired*
    aliases that should not be routed as-is.
    """

    def __init__(
        self,
        *,
        seed: dict[str, tuple[str, DeprecationSeverity, str]] | None = None,
        include_defaults: bool = True,
    ) -> None:
        """Initialize the guard with optional seed overrides.

        Args:
            seed: Optional ``model_id → (replacement, severity, reason)`` map
                merged on top of (or instead of) the built-in defaults.
            include_defaults: When ``True`` (default), start from the curated
                built-in deprecation map; when ``False``, start empty.

        Raises:
            ValueError: If a seed entry has an empty id/replacement or invalid
                severity.
        """
        self._lock = threading.Lock()
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

    def check(self, model_id: str) -> DeprecationVerdict:
        """Return a deprecation verdict for ``model_id``.

        Args:
            model_id: Requested model identifier (exact match, case-sensitive).

        Returns:
            ``DeprecationVerdict`` — ``is_deprecated`` is ``False`` when the ID
            is unknown to the guard (not an error).

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
        return DeprecationVerdict(
            model_id=model_id,
            is_deprecated=True,
            replacement=entry.replacement,
            severity=entry.severity,
            reason=entry.reason,
        )

    def register(
        self,
        model_id: str,
        replacement: str,
        *,
        severity: DeprecationSeverity = "warn",
        reason: str | None = None,
    ) -> None:
        """Register or overwrite a deprecation mapping.

        Args:
            model_id: Deprecated model identifier.
            replacement: Preferred replacement model ID.
            severity: ``warn`` (advisory) or ``block`` (reject).
            reason: Human-readable explanation (auto-generated when omitted).

        Raises:
            ValueError: If inputs are invalid.
        """
        detail = reason or f"{model_id} is deprecated; prefer {replacement}"
        self._validate(model_id, replacement, severity, detail)
        with self._lock:
            self._entries[model_id] = _DeprecationEntry(
                replacement=replacement,
                severity=severity,
                reason=detail,
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
