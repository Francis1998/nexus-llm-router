"""ContinuousBatchTokenBubbleBandAdvisor.

Advises bubble_ratio vs budgets.
Closes the vLLM/SGLang/TGI continuous-batch token-bubble monitors gap. Distinct from
`ChunkedPrefillBubbleBandAdvisor` / `DecodePreemptionPingPongBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContinuousBatchTokenBubbleBandAdvice:
    """ContinuousBatchTokenBubbleBandAdvisor advice."""

    request_id: str
    bubble_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class ContinuousBatchTokenBubbleBandAdvisor:
    """Advise bubble_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        bubble_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> ContinuousBatchTokenBubbleBandAdvice:
        """Return bubble_ratio band.

        Args:
            request_id: Non-empty request id.
            bubble_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if bubble_ratio < 0:
            raise ValueError("bubble_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if bubble_ratio >= hard_limit:
            band = "breach"
        elif bubble_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ContinuousBatchTokenBubbleBandAdvice(
            request_id=rid,
            bubble_ratio=float(bubble_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
