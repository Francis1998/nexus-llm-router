"""GpuMemoryFragmentationBandAdvisor.

Advises frag_ratio vs budgets.
Closes the vLLM/TGI/TensorRT-LLM GPU memory fragmentation monitors gap. Distinct from
`KvCacheEvictionPressureAdvisor` / `BatchCoalescePressureAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuMemoryFragmentationBandAdvice:
    """GpuMemoryFragmentationBandAdvisor advice."""

    request_id: str
    frag_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class GpuMemoryFragmentationBandAdvisor:
    """Advise frag_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        frag_ratio: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.7,
    ) -> GpuMemoryFragmentationBandAdvice:
        """Return frag_ratio band.

        Args:
            request_id: Non-empty request id.
            frag_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if frag_ratio < 0:
            raise ValueError("frag_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if frag_ratio >= hard_limit:
            band = "breach"
        elif frag_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return GpuMemoryFragmentationBandAdvice(
            request_id=rid,
            frag_ratio=float(frag_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
