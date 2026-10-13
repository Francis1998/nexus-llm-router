"""BatchPaddingWasteAdvisor.

Advises padding_waste_ratio vs budgets.
Closes the vLLM/TGI/SGLang batch-padding waste monitors gap. Distinct from
`BatchCoalescePressureAdvisor` / `EmbeddingBatchSkewAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchPaddingWasteAdvice:
    """BatchPaddingWasteAdvisor advice."""

    request_id: str
    padding_waste_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class BatchPaddingWasteAdvisor:
    """Advise padding_waste_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        padding_waste_ratio: float,
        soft_limit: float = 0.25,
        hard_limit: float = 0.55,
    ) -> BatchPaddingWasteAdvice:
        """Return padding_waste_ratio band.

        Args:
            request_id: Non-empty request id.
            padding_waste_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if padding_waste_ratio < 0:
            raise ValueError("padding_waste_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if padding_waste_ratio >= hard_limit:
            band = "breach"
        elif padding_waste_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return BatchPaddingWasteAdvice(
            request_id=rid,
            padding_waste_ratio=float(padding_waste_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
