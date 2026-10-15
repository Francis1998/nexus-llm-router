"""MoeRoutingCollapseAdvisor.

Advises collapse_rate vs budgets.
Closes the vLLM/SGLang/DeepSeek MoE routing-collapse monitors gap. Distinct from
`MoeExpertLoadImbalanceAdvisor` / `SpeculativeDecodeRejectionStormAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MoeRoutingCollapseAdvice:
    """MoeRoutingCollapseAdvisor advice."""

    request_id: str
    collapse_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class MoeRoutingCollapseAdvisor:
    """Advise collapse_rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        collapse_rate: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> MoeRoutingCollapseAdvice:
        """Return collapse_rate band.

        Args:
            request_id: Non-empty request id.
            collapse_rate: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if collapse_rate < 0:
            raise ValueError("collapse_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if collapse_rate >= hard_limit:
            band = "breach"
        elif collapse_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return MoeRoutingCollapseAdvice(
            request_id=rid,
            collapse_rate=float(collapse_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
