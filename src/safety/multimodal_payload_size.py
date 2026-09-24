"""Multimodal payload size gate.

Allows or denies requests when image/audio/file payload bytes exceed a cap.
Closes the LiteLLM / Portkey / OpenRouter multimodal payload-limit gap.
Distinct from ``ContextWindowFitGuard`` and ``OutputTokenCeilingGuard``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs
network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MultimodalPayloadDecision:
    """Payload size gate decision."""

    payload_bytes: int
    max_bytes: int
    allowed: bool
    band: str


class MultimodalPayloadSizeGate:
    """Gate oversized multimodal payloads."""

    def check(self, *, payload_bytes: int, max_bytes: int) -> MultimodalPayloadDecision:
        """Return allow/deny for multimodal payload size.

        Args:
            payload_bytes: Total attachment bytes (``>= 0``).
            max_bytes: Hard cap (``>= 1``).

        Returns:
            MultimodalPayloadDecision with band ``allow`` / ``deny``.
        """

        if payload_bytes < 0:
            raise ValueError("payload_bytes must be >= 0")
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        allowed = payload_bytes <= max_bytes
        return MultimodalPayloadDecision(
            payload_bytes=int(payload_bytes),
            max_bytes=int(max_bytes),
            allowed=allowed,
            band="allow" if allowed else "deny",
        )
