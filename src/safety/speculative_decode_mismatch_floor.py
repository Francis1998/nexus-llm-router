"""SpeculativeDecodeMismatchFloorAdvisor.

Advises mismatch_ratio vs budgets.
Closes the vLLM/SGLang speculative-decode mismatch floor monitors gap. Distinct from
`SpeculativeDecodeAcceptanceAdvisor` / `SpeculativeDecodeRejectionStormAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeDecodeMismatchFloorAdvice:
    """SpeculativeDecodeMismatchFloorAdvisor advice."""

    request_id: str
    mismatch_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class SpeculativeDecodeMismatchFloorAdvisor:
    """Advise mismatch_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        mismatch_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.55,
    ) -> SpeculativeDecodeMismatchFloorAdvice:
        """Return mismatch_ratio band.

        Args:
            request_id: Non-empty request id.
            mismatch_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if mismatch_ratio < 0:
            raise ValueError("mismatch_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if mismatch_ratio >= hard_limit:
            band = "breach"
        elif mismatch_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return SpeculativeDecodeMismatchFloorAdvice(
            request_id=rid,
            mismatch_ratio=float(mismatch_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
