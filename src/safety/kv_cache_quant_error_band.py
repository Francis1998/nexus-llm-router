"""KvCacheQuantErrorBandAdvisor.

Advises error_rate vs budgets.
Closes the vLLM/SGLang KV-cache quantization error-band monitors gap. Distinct from
`KvCacheHitRateAdvisor` / `KvBlockStealAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KvCacheQuantErrorBandAdvice:
    """KvCacheQuantErrorBandAdvisor advice."""

    request_id: str
    error_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class KvCacheQuantErrorBandAdvisor:
    """Advise error_rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        error_rate: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> KvCacheQuantErrorBandAdvice:
        """Return error_rate band.

        Args:
            request_id: Non-empty request id.
            error_rate: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if error_rate < 0:
            raise ValueError("error_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if error_rate >= hard_limit:
            band = "breach"
        elif error_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return KvCacheQuantErrorBandAdvice(
            request_id=rid,
            error_rate=float(error_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
