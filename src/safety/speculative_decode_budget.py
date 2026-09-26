"""Speculative decode budget advisor.

Advises draft-token budget bands for speculative decoding fanout.
Closes the OpenRouter / LiteLLM / Portkey speculative-decode routing gap.
Distinct from ``ReasoningTokenBudgetAdvisor`` and ``OutputTokenCeilingAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs
network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeDecodeBudgetAdvice:
    """Speculative decode budget advice."""

    request_id: str
    draft_tokens: int
    max_draft_tokens: int
    utilization: float
    band: str


class SpeculativeDecodeBudgetAdvisor:
    """Advise speculative-decode draft-token budget bands."""

    def advise(
        self,
        *,
        request_id: str,
        draft_tokens: int,
        max_draft_tokens: int = 64,
    ) -> SpeculativeDecodeBudgetAdvice:
        """Return band for draft-token utilization.

        Args:
            request_id: Non-empty request id.
            draft_tokens: Draft tokens requested (``>= 0``).
            max_draft_tokens: Draft-token ceiling (``>= 1``).

        Returns:
            SpeculativeDecodeBudgetAdvice with ``ok`` / ``elevated`` / ``saturated``.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if draft_tokens < 0:
            raise ValueError("draft_tokens must be >= 0")
        if max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be >= 1")

        util = round(min(draft_tokens / float(max_draft_tokens), 1.0), 4)
        if util >= 1.0:
            band = "saturated"
        elif util >= 0.7:
            band = "elevated"
        else:
            band = "ok"
        return SpeculativeDecodeBudgetAdvice(
            request_id=rid,
            draft_tokens=int(draft_tokens),
            max_draft_tokens=int(max_draft_tokens),
            utilization=float(util),
            band=band,
        )
