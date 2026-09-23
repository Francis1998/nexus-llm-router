"""Prompt compression ratio advisor.

Advises keep/compress/aggressive bands from prompt tokens vs context window.
Closes the LiteLLM / Portkey / OpenRouter prompt-compression gap. Distinct from
``ContextWindowFitGuard`` (hard fit) and ``OutputTokenCeilingGuard`` (output
ceiling). Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptCompressionAdvice:
    """Compression advice for one prompt."""

    prompt_tokens: int
    context_window: int
    ratio: float
    band: str


class PromptCompressionRatioAdvisor:
    """Advise prompt compression from occupancy ratio."""

    def advise(
        self,
        *,
        prompt_tokens: int,
        context_window: int,
    ) -> PromptCompressionAdvice:
        """Return compression band for ``prompt_tokens`` / ``context_window``.

        Args:
            prompt_tokens: Prompt token count (``>= 0``).
            context_window: Model context window (``> 0``).

        Returns:
            PromptCompressionAdvice with band ``keep`` / ``compress`` / ``aggressive``.
        """

        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")
        if context_window <= 0:
            raise ValueError("context_window must be > 0")

        ratio = round(prompt_tokens / context_window, 4)
        if ratio < 0.5:
            band = "keep"
        elif ratio < 0.8:
            band = "compress"
        else:
            band = "aggressive"

        return PromptCompressionAdvice(
            prompt_tokens=int(prompt_tokens),
            context_window=int(context_window),
            ratio=float(ratio),
            band=band,
        )
