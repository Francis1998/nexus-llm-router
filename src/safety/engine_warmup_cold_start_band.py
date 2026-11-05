"""EngineWarmupColdStartBandAdvisor.

Advises cold_start_ratio vs budgets.
Closes the vLLM/SGLang/TGI engine warmup cold-start monitors gap. Distinct from
`CudaGraphCaptureStallBandAdvisor` / `LoraHotSwapStallBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineWarmupColdStartBandAdvice:
    """EngineWarmupColdStartBandAdvisor advice."""

    request_id: str
    cold_start_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class EngineWarmupColdStartBandAdvisor:
    """Advise cold_start_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        cold_start_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> EngineWarmupColdStartBandAdvice:
        """Return cold_start_ratio band.

        Args:
            request_id: Non-empty request id.
            cold_start_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if cold_start_ratio < 0:
            raise ValueError("cold_start_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if cold_start_ratio >= hard_limit:
            band = "breach"
        elif cold_start_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return EngineWarmupColdStartBandAdvice(
            request_id=rid,
            cold_start_ratio=float(cold_start_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
