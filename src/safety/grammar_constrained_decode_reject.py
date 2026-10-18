"""GrammarConstrainedDecodeRejectAdvisor.

Advises reject_ratio vs budgets.
Closes the vLLM/SGLang/Outlines grammar-constrained decode reject monitors gap. Distinct from
`GrammarConstrainedDecodeBudgetAdvisor` / `JsonSchemaStrictnessBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GrammarConstrainedDecodeRejectAdvice:
    """GrammarConstrainedDecodeRejectAdvisor advice."""

    request_id: str
    reject_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class GrammarConstrainedDecodeRejectAdvisor:
    """Advise reject_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        reject_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.55,
    ) -> GrammarConstrainedDecodeRejectAdvice:
        """Return reject_ratio band.

        Args:
            request_id: Non-empty request id.
            reject_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if reject_ratio < 0:
            raise ValueError("reject_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if reject_ratio >= hard_limit:
            band = "breach"
        elif reject_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return GrammarConstrainedDecodeRejectAdvice(
            request_id=rid,
            reject_ratio=float(reject_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
