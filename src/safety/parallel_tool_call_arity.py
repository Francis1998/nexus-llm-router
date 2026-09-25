"""Parallel tool-call arity advisor.

Advises ok/busy/fanout bands for concurrent tool-call count in one turn.
Closes the OpenRouter / LiteLLM / Portkey parallel-tools fanout gap.
Distinct from ``AdaptiveConcurrencyLimiter``-style request concurrency and
``RequestHedgingAdvisor``. Works with GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2. Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParallelToolCallArityAdvice:
    """Parallel tool-call arity advice."""

    tool_call_count: int
    soft_limit: int
    hard_limit: int
    band: str


class ParallelToolCallArityAdvisor:
    """Advise parallel tool-call arity bands."""

    def advise(
        self,
        *,
        tool_call_count: int,
        soft_limit: int = 4,
        hard_limit: int = 12,
    ) -> ParallelToolCallArityAdvice:
        """Return band for parallel tool-call count.

        Args:
            tool_call_count: Tool calls in this turn (``>= 0``).
            soft_limit: Soft advisory threshold (``>= 1``).
            hard_limit: Hard fanout threshold (``>= soft_limit``).

        Returns:
            ParallelToolCallArityAdvice with ``ok`` / ``busy`` / ``fanout``.
        """

        if tool_call_count < 0:
            raise ValueError("tool_call_count must be >= 0")
        if soft_limit < 1:
            raise ValueError("soft_limit must be >= 1")
        if hard_limit < soft_limit:
            raise ValueError("hard_limit must be >= soft_limit")

        if tool_call_count >= hard_limit:
            band = "fanout"
        elif tool_call_count >= soft_limit:
            band = "busy"
        else:
            band = "ok"
        return ParallelToolCallArityAdvice(
            tool_call_count=int(tool_call_count),
            soft_limit=int(soft_limit),
            hard_limit=int(hard_limit),
            band=band,
        )
