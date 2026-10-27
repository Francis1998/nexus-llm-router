"""PagedAttentionFragmentationBandAdvisor.

Advises fragmentation_ratio vs budgets.
Closes the vLLM/SGLang/TGI paged-attention fragmentation monitors gap. Distinct from
`KvBlockStealStormBandAdvisor` / `ChunkedPrefillBubbleBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PagedAttentionFragmentationBandAdvice:
    """PagedAttentionFragmentationBandAdvisor advice."""

    request_id: str
    fragmentation_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class PagedAttentionFragmentationBandAdvisor:
    """Advise fragmentation_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        fragmentation_ratio: float,
        soft_limit: float = 0.25,
        hard_limit: float = 0.55,
    ) -> PagedAttentionFragmentationBandAdvice:
        """Return fragmentation_ratio band.

        Args:
            request_id: Non-empty request id.
            fragmentation_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if fragmentation_ratio < 0:
            raise ValueError("fragmentation_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if fragmentation_ratio >= hard_limit:
            band = "breach"
        elif fragmentation_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return PagedAttentionFragmentationBandAdvice(
            request_id=rid,
            fragmentation_ratio=float(fragmentation_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
