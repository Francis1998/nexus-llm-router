"""SpeculativeDecodeRejectionStorm advisor.

Advises rejection_rate vs budgets.
Closes the vLLM/SGLang/TensorRT-LLM speculative-decode rejection-storm advisors gap. Distinct from
``SpeculativeDecodeAcceptanceBandAdvisor`` and ``SpeculativeDecodeAbortAdvisor``.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeDecodeRejectionStormAdvice:
    """SpeculativeDecodeRejectionStormAdvisor advice."""

    request_id: str
    rejection_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class SpeculativeDecodeRejectionStormAdvisor:
    """Advise rejection_rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        rejection_rate: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.6,
    ) -> SpeculativeDecodeRejectionStormAdvice:
        """Return rejection_rate band.

        Args:
            request_id: Non-empty request id.
            rejection_rate: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if rejection_rate < 0:
            raise ValueError("rejection_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if rejection_rate >= hard_limit:
            band = "breach"
        elif rejection_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return SpeculativeDecodeRejectionStormAdvice(
            request_id=rid,
            rejection_rate=float(rejection_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
