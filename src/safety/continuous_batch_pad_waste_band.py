"""ContinuousBatchPadWasteBandAdvisor.

Advises pad_waste_ratio vs budgets.
Closes the vLLM/SGLang/TGI continuous-batch pad-waste monitors gap. Distinct from
`ContinuousBatchTokenBubbleBandAdvisor` / `ContinuousBatchingTailLatencyBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContinuousBatchPadWasteBandAdvice:
    """ContinuousBatchPadWasteBandAdvisor advice."""

    request_id: str
    pad_waste_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class ContinuousBatchPadWasteBandAdvisor:
    """Advise pad_waste_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        pad_waste_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> ContinuousBatchPadWasteBandAdvice:
        """Return pad_waste_ratio band.

        Args:
            request_id: Non-empty request id.
            pad_waste_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if pad_waste_ratio < 0:
            raise ValueError("pad_waste_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if pad_waste_ratio >= hard_limit:
            band = "breach"
        elif pad_waste_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ContinuousBatchPadWasteBandAdvice(
            request_id=rid,
            pad_waste_ratio=float(pad_waste_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
