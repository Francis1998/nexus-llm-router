"""KvBlockStealStormBandAdvisor.

Advises steal_ratio vs budgets.
Closes the vLLM/SGLang/TGI KV block steal-storm monitors gap. Distinct from
`KvBlockStealAdvisor` / `PrefixCacheEvictionStormAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KvBlockStealStormBandAdvice:
    """KvBlockStealStormBandAdvisor advice."""

    request_id: str
    steal_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class KvBlockStealStormBandAdvisor:
    """Advise steal_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        steal_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> KvBlockStealStormBandAdvice:
        """Return steal_ratio band.

        Args:
            request_id: Non-empty request id.
            steal_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if steal_ratio < 0:
            raise ValueError("steal_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if steal_ratio >= hard_limit:
            band = "breach"
        elif steal_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return KvBlockStealStormBandAdvice(
            request_id=rid,
            steal_ratio=float(steal_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
