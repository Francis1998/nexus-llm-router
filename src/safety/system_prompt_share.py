"""System-prompt share-of-context advisor.

Advises ok/heavy/bloated bands for system-prompt token share of the context
window. Closes the OpenRouter / LiteLLM / Portkey system-prompt bloat gap.
Distinct from ``ContextWindowFitAdvisor`` and ``PromptCompressionRatioAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs
network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemPromptShareAdvice:
    """System-prompt share advice."""

    system_prompt_tokens: int
    context_window_tokens: int
    share_ratio: float
    band: str


class SystemPromptShareAdvisor:
    """Advise system-prompt share-of-context bands."""

    def advise(
        self,
        *,
        system_prompt_tokens: int,
        context_window_tokens: int,
    ) -> SystemPromptShareAdvice:
        """Return band for system-prompt share of context.

        Args:
            system_prompt_tokens: System prompt tokens (``>= 0``).
            context_window_tokens: Model context window (``>= 1``).

        Returns:
            SystemPromptShareAdvice with ``ok`` / ``heavy`` / ``bloated``.
        """

        if system_prompt_tokens < 0:
            raise ValueError("system_prompt_tokens must be >= 0")
        if context_window_tokens < 1:
            raise ValueError("context_window_tokens must be >= 1")
        if system_prompt_tokens > context_window_tokens:
            raise ValueError("system_prompt_tokens must be <= context_window_tokens")

        share = round(system_prompt_tokens / context_window_tokens, 4)
        if share >= 0.35:
            band = "bloated"
        elif share >= 0.2:
            band = "heavy"
        else:
            band = "ok"
        return SystemPromptShareAdvice(
            system_prompt_tokens=int(system_prompt_tokens),
            context_window_tokens=int(context_window_tokens),
            share_ratio=float(share),
            band=band,
        )
