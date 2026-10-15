"""KvBlockStealAdvisor.

Advises steal_rate vs budgets.
Closes the vLLM/SGLang/TGI KV-block steal monitors gap. Distinct from
`KvCacheEvictionPressureAdvisor` / `KvCacheHitRateAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KvBlockStealAdvice:
    """KvBlockStealAdvisor advice."""

    request_id: str
    steal_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class KvBlockStealAdvisor:
    """Advise steal_rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        steal_rate: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> KvBlockStealAdvice:
        """Return steal_rate band.

        Args:
            request_id: Non-empty request id.
            steal_rate: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if steal_rate < 0:
            raise ValueError("steal_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if steal_rate >= hard_limit:
            band = "breach"
        elif steal_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return KvBlockStealAdvice(
            request_id=rid,
            steal_rate=float(steal_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
