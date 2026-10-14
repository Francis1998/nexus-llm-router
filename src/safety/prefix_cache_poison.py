"""PrefixCachePoisonAdvisor.

Advises poison_hit_rate vs budgets.
Closes the vLLM/SGLang/TGI prefix-cache poison monitors gap. Distinct from
`PrefixCacheThrashAdvisor` / `KvCacheHitRateAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefixCachePoisonAdvice:
    """PrefixCachePoisonAdvisor advice."""

    request_id: str
    poison_hit_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class PrefixCachePoisonAdvisor:
    """Advise poison_hit_rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        poison_hit_rate: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> PrefixCachePoisonAdvice:
        """Return poison_hit_rate band.

        Args:
            request_id: Non-empty request id.
            poison_hit_rate: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if poison_hit_rate < 0:
            raise ValueError("poison_hit_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if poison_hit_rate >= hard_limit:
            band = "breach"
        elif poison_hit_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return PrefixCachePoisonAdvice(
            request_id=rid,
            poison_hit_rate=float(poison_hit_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
