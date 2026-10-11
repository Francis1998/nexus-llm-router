"""RegionFailoverHysteresisAdvisor.

Advises failover_score vs budgets.
Closes the LiteLLM/OpenRouter region-failover hysteresis monitors gap. Distinct from
`ProviderRegionAffinityAdvisor` and `ProviderHealthHysteresisAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegionFailoverHysteresisAdvice:
    """RegionFailoverHysteresisAdvisor advice."""

    request_id: str
    failover_score: float
    soft_limit: float
    hard_limit: float
    band: str


class RegionFailoverHysteresisAdvisor:
    """Advise failover_score bands."""

    def advise(
        self,
        *,
        request_id: str,
        failover_score: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.7,
    ) -> RegionFailoverHysteresisAdvice:
        """Return failover_score band.

        Args:
            request_id: Non-empty request id.
            failover_score: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if failover_score < 0:
            raise ValueError("failover_score must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if failover_score >= hard_limit:
            band = "breach"
        elif failover_score >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return RegionFailoverHysteresisAdvice(
            request_id=rid,
            failover_score=float(failover_score),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
