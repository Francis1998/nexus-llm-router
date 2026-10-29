"""ExpertParallelImbalanceBandAdvisor.

Advises imbalance_ratio vs budgets.
Closes the vLLM/SGLang/TGI expert-parallel (MoE) imbalance monitors gap. Distinct from
`TensorParallelSkewBandAdvisor` / `ContinuousBatchTokenBubbleBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpertParallelImbalanceBandAdvice:
    """ExpertParallelImbalanceBandAdvisor advice."""

    request_id: str
    imbalance_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class ExpertParallelImbalanceBandAdvisor:
    """Advise imbalance_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        imbalance_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> ExpertParallelImbalanceBandAdvice:
        """Return imbalance_ratio band.

        Args:
            request_id: Non-empty request id.
            imbalance_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if imbalance_ratio < 0:
            raise ValueError("imbalance_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if imbalance_ratio >= hard_limit:
            band = "breach"
        elif imbalance_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ExpertParallelImbalanceBandAdvice(
            request_id=rid,
            imbalance_ratio=float(imbalance_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
