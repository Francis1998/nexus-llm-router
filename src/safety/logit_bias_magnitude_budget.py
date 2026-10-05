"""LogitBiasMagnitudeBudget advisor.

Advises l1_magnitude vs budgets.
Closes the OpenAI/vLLM/LiteLLM logit-bias magnitude budget guards gap. Distinct from
``GrammarConstrainedDecodeBudgetAdvisor`` and ``OutputTokenCeilingGuard``.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LogitBiasMagnitudeBudgetAdvice:
    """LogitBiasMagnitudeBudgetGuard advice."""

    request_id: str
    l1_magnitude: float
    soft_limit: float
    hard_limit: float
    band: str


class LogitBiasMagnitudeBudgetGuard:
    """Advise l1_magnitude bands."""

    def advise(
        self,
        *,
        request_id: str,
        l1_magnitude: float,
        soft_limit: float = 8.0,
        hard_limit: float = 24.0,
    ) -> LogitBiasMagnitudeBudgetAdvice:
        """Return l1_magnitude band.

        Args:
            request_id: Non-empty request id.
            l1_magnitude: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget (``> 0``).
            hard_limit: Hard budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if l1_magnitude < 0:
            raise ValueError("l1_magnitude must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if l1_magnitude >= hard_limit:
            band = "breach"
        elif l1_magnitude >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return LogitBiasMagnitudeBudgetAdvice(
            request_id=rid,
            l1_magnitude=float(l1_magnitude),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
