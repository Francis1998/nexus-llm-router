"""PagedAttentionBlockLeakBandAdvisor.

Advises leak_ratio vs budgets.
Closes the vLLM/SGLang/TGI paged-attention block-leak monitors gap. Distinct from
`PagedAttentionFragmentationBandAdvisor` / `KvBlockStealStormBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PagedAttentionBlockLeakBandAdvice:
    """PagedAttentionBlockLeakBandAdvisor advice."""

    request_id: str
    leak_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class PagedAttentionBlockLeakBandAdvisor:
    """Advise leak_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        leak_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> PagedAttentionBlockLeakBandAdvice:
        """Return leak_ratio band.

        Args:
            request_id: Non-empty request id.
            leak_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if leak_ratio < 0:
            raise ValueError("leak_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if leak_ratio >= hard_limit:
            band = "breach"
        elif leak_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return PagedAttentionBlockLeakBandAdvice(
            request_id=rid,
            leak_ratio=float(leak_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
