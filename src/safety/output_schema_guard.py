"""Validate model JSON output against a simple required-keys schema."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

SchemaVerdict = Literal["ok", "missing", "invalid_json"]


@dataclass(frozen=True, slots=True)
class OutputSchemaVerdict:
    """Outcome of checking model text against a required-keys schema."""

    verdict: SchemaVerdict
    missing_keys: list[str] = field(default_factory=list)
    parsed: dict[str, Any] | None = None
    reason: str | None = None


class OutputJsonSchemaGuard:
    """Validate model JSON output for required top-level keys.

    Closes the LiteLLM / Portkey / OpenRouter structured-output *post-check*
    gap for offline Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2: after a completion returns text, ``check(text)``
    parses JSON and verifies every configured required key is present.

    Distinct from ``PromptInjectionGateway`` (pre-route injection allow /
    redact / block) and from ``PiiRedactor`` / PII controls (sensitive-data
    scrubbing). This guard only validates JSON shape via required keys —
    verdict ``ok`` / ``missing`` / ``invalid_json``.
    """

    def __init__(self, *, required_keys: list[str]) -> None:
        """Configure the required top-level JSON object keys.

        Args:
            required_keys: Non-empty list of unique non-empty key names that
                must appear in a parsed JSON object.

        Raises:
            ValueError: If the list is empty, contains blanks, or duplicates.
        """
        if not required_keys:
            raise ValueError("required_keys must be non-empty")
        cleaned: list[str] = []
        seen: set[str] = set()
        for key in required_keys:
            if not key:
                raise ValueError("required_keys entries must be non-empty")
            if key in seen:
                raise ValueError(f"duplicate required key: {key}")
            seen.add(key)
            cleaned.append(key)
        self._required_keys = cleaned

    @property
    def required_keys(self) -> list[str]:
        """Return a copy of the configured required key list."""
        return list(self._required_keys)

    def check(self, text: str) -> OutputSchemaVerdict:
        """Parse ``text`` as JSON and verify required keys.

        Args:
            text: Raw model output expected to be a JSON object string.

        Returns:
            ``OutputSchemaVerdict`` with ``ok``, ``missing``, or
            ``invalid_json``.
        """
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return OutputSchemaVerdict(
                verdict="invalid_json",
                missing_keys=[],
                parsed=None,
                reason="payload is not valid JSON",
            )
        if not isinstance(parsed, dict):
            return OutputSchemaVerdict(
                verdict="invalid_json",
                missing_keys=[],
                parsed=None,
                reason="JSON root must be an object",
            )
        missing = [key for key in self._required_keys if key not in parsed]
        if missing:
            return OutputSchemaVerdict(
                verdict="missing",
                missing_keys=missing,
                parsed=parsed,
                reason=f"missing required keys: {', '.join(missing)}",
            )
        return OutputSchemaVerdict(
            verdict="ok",
            missing_keys=[],
            parsed=parsed,
            reason=None,
        )
