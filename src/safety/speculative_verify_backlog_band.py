"""SpeculativeVerifyBacklogBandAdvisor.

Advises backlog_ratio vs budgets.
Closes the vLLM/SGLang/TGI speculative verify backlog monitors gap. Distinct from
`SpeculativeDecodeQueueStallAdvisor` / `SpeculativeDraftStarveBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeVerifyBacklogBandAdvice:
    """SpeculativeVerifyBacklogBandAdvisor advice."""

    request_id: str
    backlog_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class SpeculativeVerifyBacklogBandAdvisor:
    """Advise backlog_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        backlog_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> SpeculativeVerifyBacklogBandAdvice:
        """Return backlog_ratio band.

        Args:
            request_id: Non-empty request id.
            backlog_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if backlog_ratio < 0:
            raise ValueError("backlog_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if backlog_ratio >= hard_limit:
            band = "breach"
        elif backlog_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return SpeculativeVerifyBacklogBandAdvice(
            request_id=rid,
            backlog_ratio=float(backlog_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
