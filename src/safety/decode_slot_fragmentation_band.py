"""DecodeSlotFragmentationBandAdvisor.

Advises slot_frag_ratio vs budgets.
Closes the vLLM/SGLang/TGI decode-slot fragmentation monitors gap. Distinct from
`PagedAttentionSlotFragmentationAdvisor` / `GpuMemoryFragmentationBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecodeSlotFragmentationBandAdvice:
    """DecodeSlotFragmentationBandAdvisor advice."""

    request_id: str
    slot_frag_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class DecodeSlotFragmentationBandAdvisor:
    """Advise slot_frag_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        slot_frag_ratio: float,
        soft_limit: float = 0.28,
        hard_limit: float = 0.65,
    ) -> DecodeSlotFragmentationBandAdvice:
        """Return slot_frag_ratio band.

        Args:
            request_id: Non-empty request id.
            slot_frag_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if slot_frag_ratio < 0:
            raise ValueError("slot_frag_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if slot_frag_ratio >= hard_limit:
            band = "breach"
        elif slot_frag_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return DecodeSlotFragmentationBandAdvice(
            request_id=rid,
            slot_frag_ratio=float(slot_frag_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
