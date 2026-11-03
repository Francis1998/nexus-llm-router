"""SlidingWindowAttentionBleedBandAdvisor.

Advises bleed_ratio vs budgets.
Closes the vLLM/SGLang/TGI sliding-window attention bleed monitors gap. Distinct from
`AttentionSinkResidualBandAdvisor` / `RopeScaleMismatchBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SlidingWindowAttentionBleedBandAdvice:
    """SlidingWindowAttentionBleedBandAdvisor advice."""

    request_id: str
    bleed_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class SlidingWindowAttentionBleedBandAdvisor:
    """Advise bleed_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        bleed_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> SlidingWindowAttentionBleedBandAdvice:
        """Return bleed_ratio band.

        Args:
            request_id: Non-empty request id.
            bleed_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if bleed_ratio < 0:
            raise ValueError("bleed_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if bleed_ratio >= hard_limit:
            band = "breach"
        elif bleed_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return SlidingWindowAttentionBleedBandAdvice(
            request_id=rid,
            bleed_ratio=float(bleed_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
