"""RadixCacheCollisionBandAdvisor.

Advises collision_ratio vs budgets.
Closes the vLLM/SGLang/TGI radix-cache collision monitors gap. Distinct from
`PrefixCacheEvictionStormAdvisor` / `KvBlockStealStormBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RadixCacheCollisionBandAdvice:
    """RadixCacheCollisionBandAdvisor advice."""

    request_id: str
    collision_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class RadixCacheCollisionBandAdvisor:
    """Advise collision_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        collision_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> RadixCacheCollisionBandAdvice:
        """Return collision_ratio band.

        Args:
            request_id: Non-empty request id.
            collision_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if collision_ratio < 0:
            raise ValueError("collision_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if collision_ratio >= hard_limit:
            band = "breach"
        elif collision_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return RadixCacheCollisionBandAdvice(
            request_id=rid,
            collision_ratio=float(collision_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
