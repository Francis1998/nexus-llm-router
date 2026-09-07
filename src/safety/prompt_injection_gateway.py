"""Prompt-injection detection gateway with allow / redact / block actions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InjectionVerdict:
    """Result of inspecting text for classic prompt-injection patterns."""

    risk: float
    action: str  # allow|redact|block
    matched_patterns: tuple[str, ...]
    sanitized_text: str | None


class PromptInjectionBlockedError(RuntimeError):
    """Raised when ``enforce`` decides the text must be blocked."""


# Classic jailbreak / instruction-override patterns (case-insensitive).
_PATTERNS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    (
        "ignore_previous_instructions",
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
            re.I,
        ),
        0.95,
    ),
    (
        "disregard_system_prompt",
        re.compile(r"disregard\s+(the\s+)?(system\s+)?(prompt|instructions)", re.I),
        0.9,
    ),
    (
        "system_prompt_override",
        re.compile(
            r"(reveal|show|print|dump)\s+(your\s+)?(system\s+prompt|hidden\s+instructions)",
            re.I,
        ),
        0.88,
    ),
    (
        "system_prompt_mention",
        re.compile(r"\bsystem\s+prompt\b", re.I),
        0.55,
    ),
    (
        "jailbreak",
        re.compile(r"\bjailbreak\b", re.I),
        0.8,
    ),
    (
        "dan_mode",
        re.compile(r"\bDAN\s+mode\b|\byou\s+are\s+DAN\b|\bdo\s+anything\s+now\b", re.I),
        0.92,
    ),
    (
        "developer_mode",
        re.compile(r"\bdeveloper\s+mode\b|\benable\s+developer\s+mode\b", re.I),
        0.75,
    ),
    (
        "bypass_safety",
        re.compile(r"bypass\s+(safety|filters?|guardrails?|content\s+policy)", re.I),
        0.85,
    ),
    (
        "pretend_unrestricted",
        re.compile(
            r"(pretend|act\s+as\s+if)\s+you\s+(have\s+no|are\s+without)\s+(restrictions|limits|rules)",
            re.I,
        ),
        0.8,
    ),
    (
        "new_instructions_override",
        re.compile(r"new\s+instructions?\s*:\s*", re.I),
        0.7,
    ),
)


class PromptInjectionGateway:
    """Detect classic prompt-injection patterns and allow / redact / block.

    Distinct from the ``prompt-injection-risk-shed`` *routing* strategy, which
    only demotes high-risk traffic to cheaper models. This gateway inspects
    text and can sanitize or refuse before dispatch.
    """

    def __init__(
        self,
        *,
        block_threshold: float = 0.85,
        redact_threshold: float = 0.55,
    ) -> None:
        """Initialize risk thresholds.

        Args:
            block_threshold: Risk at or above this value blocks (default ``0.85``).
            redact_threshold: Risk at or above this (and below block) redacts
                matched spans (default ``0.55``). Below redact is allow.
        """
        if not 0.0 <= redact_threshold <= block_threshold <= 1.0:
            raise ValueError("require 0 <= redact_threshold <= block_threshold <= 1")
        self._block_threshold = float(block_threshold)
        self._redact_threshold = float(redact_threshold)

    @property
    def block_threshold(self) -> float:
        """Return the configured block threshold."""
        return self._block_threshold

    @property
    def redact_threshold(self) -> float:
        """Return the configured redact threshold."""
        return self._redact_threshold

    def inspect(self, text: str) -> InjectionVerdict:
        """Score ``text`` and decide allow / redact / block.

        Args:
            text: User or tool text to inspect.

        Returns:
            ``InjectionVerdict`` with risk, action, matches, and optional sanitize.
        """
        matched: list[str] = []
        risk = 0.0
        for name, pattern, weight in _PATTERNS:
            if pattern.search(text):
                matched.append(name)
                risk = max(risk, weight)
        matched_tuple = tuple(matched)
        if risk >= self._block_threshold:
            return InjectionVerdict(
                risk=risk,
                action="block",
                matched_patterns=matched_tuple,
                sanitized_text=None,
            )
        if risk >= self._redact_threshold:
            sanitized = text
            for _name, pattern, _weight in _PATTERNS:
                sanitized = pattern.sub("[REDACTED_INJECTION]", sanitized)
            return InjectionVerdict(
                risk=risk,
                action="redact",
                matched_patterns=matched_tuple,
                sanitized_text=sanitized,
            )
        return InjectionVerdict(
            risk=risk,
            action="allow",
            matched_patterns=matched_tuple,
            sanitized_text=None,
        )

    def enforce(self, text: str) -> str:
        """Return allowed or sanitized text; raise on block.

        Args:
            text: User or tool text to enforce policy on.

        Returns:
            Original text (allow) or sanitized text (redact).

        Raises:
            PromptInjectionBlockedError: When the verdict action is ``block``.
        """
        verdict = self.inspect(text)
        if verdict.action == "block":
            patterns = ", ".join(verdict.matched_patterns) or "unknown"
            raise PromptInjectionBlockedError(
                f"prompt injection blocked (risk={verdict.risk:.2f}; patterns={patterns})"
            )
        if verdict.action == "redact":
            assert verdict.sanitized_text is not None
            return verdict.sanitized_text
        return text
