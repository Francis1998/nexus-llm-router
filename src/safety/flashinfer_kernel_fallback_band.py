"""FlashinferKernelFallbackBandAdvisor.

Advises fallback_ratio vs budgets.
Closes the vLLM/SGLang/TGI FlashInfer kernel-fallback monitors gap. Distinct from
`KvCacheQuantErrorBandAdvisor` / `TensorParallelSkewBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FlashinferKernelFallbackBandAdvice:
    """FlashinferKernelFallbackBandAdvisor advice."""

    request_id: str
    fallback_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class FlashinferKernelFallbackBandAdvisor:
    """Advise fallback_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        fallback_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> FlashinferKernelFallbackBandAdvice:
        """Return fallback_ratio band.

        Args:
            request_id: Non-empty request id.
            fallback_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if fallback_ratio < 0:
            raise ValueError("fallback_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if fallback_ratio >= hard_limit:
            band = "breach"
        elif fallback_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return FlashinferKernelFallbackBandAdvice(
            request_id=rid,
            fallback_ratio=float(fallback_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
