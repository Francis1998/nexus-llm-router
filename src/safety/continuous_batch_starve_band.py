"""ContinuousBatchStarveBandAdvisor.

Advises starve_ratio vs budgets.
Closes the vLLM/SGLang/TGI continuous-batch starve monitors gap. Distinct from
`ContinuousBatchPreemptionChurnAdvisor` / `ContinuousBatchingTailLatencyBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContinuousBatchStarveBandAdvice:
    """ContinuousBatchStarveBandAdvisor advice."""

    request_id: str
    starve_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class ContinuousBatchStarveBandAdvisor:
    """Advise starve_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        starve_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> ContinuousBatchStarveBandAdvice:
        """Return starve_ratio band.

        Args:
            request_id: Non-empty request id.
            starve_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if starve_ratio < 0:
            raise ValueError("starve_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if starve_ratio >= hard_limit:
            band = "breach"
        elif starve_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ContinuousBatchStarveBandAdvice(
            request_id=rid,
            starve_ratio=float(starve_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
