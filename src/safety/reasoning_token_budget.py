"""Reasoning / thinking-token budget advisor.

Advises ok/near_limit/exhausted bands for reasoning-token usage (GPT-5.5
thinking / Claude extended thinking style workloads). Closes the OpenRouter /
LiteLLM / Portkey reasoning-budget gap. Distinct from ``StreamBudgetGuard`` and
``OutputTokenCeilingGuard``. Works with GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2. Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReasoningTokenBudgetAdvice:
    """Reasoning-token budget advice."""

    reasoning_tokens_used: int
    reasoning_token_budget: int
    remaining: int
    band: str


class ReasoningTokenBudgetAdvisor:
    """Advise reasoning-token budget bands."""

    def advise(
        self,
        *,
        reasoning_tokens_used: int,
        reasoning_token_budget: int,
    ) -> ReasoningTokenBudgetAdvice:
        """Return band for reasoning token usage.

        Args:
            reasoning_tokens_used: Thinking tokens consumed (``>= 0``).
            reasoning_token_budget: Hard reasoning budget (``>= 1``).

        Returns:
            ReasoningTokenBudgetAdvice with ``ok`` / ``near_limit`` / ``exhausted``.
        """

        if reasoning_tokens_used < 0:
            raise ValueError("reasoning_tokens_used must be >= 0")
        if reasoning_token_budget < 1:
            raise ValueError("reasoning_token_budget must be >= 1")
        remaining = max(0, reasoning_token_budget - reasoning_tokens_used)
        if remaining == 0:
            band = "exhausted"
        elif remaining <= max(1, reasoning_token_budget // 10):
            band = "near_limit"
        else:
            band = "ok"
        return ReasoningTokenBudgetAdvice(
            reasoning_tokens_used=int(reasoning_tokens_used),
            reasoning_token_budget=int(reasoning_token_budget),
            remaining=int(remaining),
            band=band,
        )
