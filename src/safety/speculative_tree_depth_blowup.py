"""SpeculativeTreeDepthBlowupAdvisor.

Advises depth_blowup_ratio vs budgets.
Closes the vLLM/SGLang/TGI speculative tree-depth blowup monitors gap. Distinct from
`SpeculativeDecodeRejectionHysteresisAdvisor` / `GuidedDecodeFsmBlowupAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeTreeDepthBlowupAdvice:
    """SpeculativeTreeDepthBlowupAdvisor advice."""

    request_id: str
    depth_blowup_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class SpeculativeTreeDepthBlowupAdvisor:
    """Advise depth_blowup_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        depth_blowup_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> SpeculativeTreeDepthBlowupAdvice:
        """Return depth_blowup_ratio band.

        Args:
            request_id: Non-empty request id.
            depth_blowup_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if depth_blowup_ratio < 0:
            raise ValueError("depth_blowup_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if depth_blowup_ratio >= hard_limit:
            band = "breach"
        elif depth_blowup_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return SpeculativeTreeDepthBlowupAdvice(
            request_id=rid,
            depth_blowup_ratio=float(depth_blowup_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
