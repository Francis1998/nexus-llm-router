"""DecodeBatchPriorityInversionAdvisor.

Advises inversion_ratio vs budgets.
Closes the vLLM/SGLang/TGI decode-batch priority-inversion monitors gap. Distinct from
`PrefillQueueStarvationAdvisor` / `KvBlockStealAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecodeBatchPriorityInversionAdvice:
    """DecodeBatchPriorityInversionAdvisor advice."""

    request_id: str
    inversion_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class DecodeBatchPriorityInversionAdvisor:
    """Advise inversion_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        inversion_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.55,
    ) -> DecodeBatchPriorityInversionAdvice:
        """Return inversion_ratio band.

        Args:
            request_id: Non-empty request id.
            inversion_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if inversion_ratio < 0:
            raise ValueError("inversion_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if inversion_ratio >= hard_limit:
            band = "breach"
        elif inversion_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return DecodeBatchPriorityInversionAdvice(
            request_id=rid,
            inversion_ratio=float(inversion_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
