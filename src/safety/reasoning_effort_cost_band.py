"""ReasoningEffortCostBand advisor.

Advises effort_cost_usd vs budgets.
Closes the OpenAI/Anthropic/Gemini reasoning-effort cost-band advisors gap. Distinct from
``ReasoningTokenBudgetAdvisor`` and ``GrammarConstrainedDecodeBudgetAdvisor``.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReasoningEffortCostBandAdvice:
    """ReasoningEffortCostBandAdvisor advice."""

    request_id: str
    effort_cost_usd: float
    soft_limit: float
    hard_limit: float
    band: str


class ReasoningEffortCostBandAdvisor:
    """Advise effort_cost_usd bands."""

    def advise(
        self,
        *,
        request_id: str,
        effort_cost_usd: float,
        soft_limit: float = 0.05,
        hard_limit: float = 0.25,
    ) -> ReasoningEffortCostBandAdvice:
        """Return effort_cost_usd band.

        Args:
            request_id: Non-empty request id.
            effort_cost_usd: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget (``> 0``).
            hard_limit: Hard budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if effort_cost_usd < 0:
            raise ValueError("effort_cost_usd must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if effort_cost_usd >= hard_limit:
            band = "breach"
        elif effort_cost_usd >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ReasoningEffortCostBandAdvice(
            request_id=rid,
            effort_cost_usd=float(effort_cost_usd),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
