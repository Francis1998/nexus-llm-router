"""SpeculativeDecodeQueueStallAdvisor.

Advises stall_ratio vs budgets.
Closes the vLLM/TGI/OpenAI-compatible speculative-decode queue-stall monitors gap. Distinct from
`SpeculativeDecodeRejectionStormAdvisor` and `SpeculativeDecodeAbortAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeDecodeQueueStallAdvice:
    """SpeculativeDecodeQueueStallAdvisor advice."""

    request_id: str
    stall_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class SpeculativeDecodeQueueStallAdvisor:
    """Advise stall_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        stall_ratio: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.7,
    ) -> SpeculativeDecodeQueueStallAdvice:
        """Return stall_ratio band.

        Args:
            request_id: Non-empty request id.
            stall_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if stall_ratio < 0:
            raise ValueError("stall_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if stall_ratio >= hard_limit:
            band = "breach"
        elif stall_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return SpeculativeDecodeQueueStallAdvice(
            request_id=rid,
            stall_ratio=float(stall_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
