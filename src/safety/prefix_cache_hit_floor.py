"""PrefixCacheHitFloorAdvisor.

Advises miss_ratio vs budgets.
Closes the vLLM/SGLang/TGI prefix-cache hit floor monitors gap. Distinct from
`PrefixCacheThrashAdvisor` / `PrefixCachePoisonAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefixCacheHitFloorAdvice:
    """PrefixCacheHitFloorAdvisor advice."""

    request_id: str
    miss_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class PrefixCacheHitFloorAdvisor:
    """Advise miss_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        miss_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.55,
    ) -> PrefixCacheHitFloorAdvice:
        """Return miss_ratio band.

        Args:
            request_id: Non-empty request id.
            miss_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if miss_ratio < 0:
            raise ValueError("miss_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if miss_ratio >= hard_limit:
            band = "breach"
        elif miss_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return PrefixCacheHitFloorAdvice(
            request_id=rid,
            miss_ratio=float(miss_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
