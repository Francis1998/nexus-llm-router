"""GrammarConstrainedDecodeBudget advisor.

Advises constraint_tokens vs budgets.
Closes the vLLM/Outlines/XGrammar grammar-constrained decode budget advisors gap. Distinct from
``StructuredOutputRepairAdvisor`` and ``OutputTokenCeilingGuard``.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GrammarConstrainedDecodeBudgetAdvice:
    """GrammarConstrainedDecodeBudgetAdvisor advice."""

    request_id: str
    constraint_tokens: float
    soft_limit: float
    hard_limit: float
    band: str


class GrammarConstrainedDecodeBudgetAdvisor:
    """Advise constraint_tokens bands."""

    def advise(
        self,
        *,
        request_id: str,
        constraint_tokens: float,
        soft_limit: float = 256.0,
        hard_limit: float = 1024.0,
    ) -> GrammarConstrainedDecodeBudgetAdvice:
        """Return constraint_tokens band.

        Args:
            request_id: Non-empty request id.
            constraint_tokens: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget (``> 0``).
            hard_limit: Hard budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if constraint_tokens < 0:
            raise ValueError("constraint_tokens must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if constraint_tokens >= hard_limit:
            band = "breach"
        elif constraint_tokens >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return GrammarConstrainedDecodeBudgetAdvice(
            request_id=rid,
            constraint_tokens=float(constraint_tokens),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
