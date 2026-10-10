"""BatchCoalescePressureAdvisor.

Advises coalesce_pressure vs budgets.
Closes the vLLM/TGI/OpenAI-compatible batch-coalesce pressure monitors gap. Distinct from
`EmbeddingBatchSkewAdvisor` and `TokenBucketBurstStrategy`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchCoalescePressureAdvice:
    """BatchCoalescePressureAdvisor advice."""

    request_id: str
    coalesce_pressure: float
    soft_limit: float
    hard_limit: float
    band: str


class BatchCoalescePressureAdvisor:
    """Advise coalesce_pressure bands."""

    def advise(
        self,
        *,
        request_id: str,
        coalesce_pressure: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.7,
    ) -> BatchCoalescePressureAdvice:
        """Return coalesce_pressure band.

        Args:
            request_id: Non-empty request id.
            coalesce_pressure: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if coalesce_pressure < 0:
            raise ValueError("coalesce_pressure must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if coalesce_pressure >= hard_limit:
            band = "breach"
        elif coalesce_pressure >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return BatchCoalescePressureAdvice(
            request_id=rid,
            coalesce_pressure=float(coalesce_pressure),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
