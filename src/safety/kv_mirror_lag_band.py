"""KvMirrorLagBandAdvisor.

Advises lag_ratio vs budgets.
Closes the vLLM/SGLang/TGI KV-mirror lag monitors gap. Distinct from
`DisaggKvTransferStallBandAdvisor` / `KvCacheOffloadThrashBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KvMirrorLagBandAdvice:
    """KvMirrorLagBandAdvisor advice."""

    request_id: str
    lag_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class KvMirrorLagBandAdvisor:
    """Advise lag_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        lag_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> KvMirrorLagBandAdvice:
        """Return lag_ratio band.

        Args:
            request_id: Non-empty request id.
            lag_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if lag_ratio < 0:
            raise ValueError("lag_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if lag_ratio >= hard_limit:
            band = "breach"
        elif lag_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return KvMirrorLagBandAdvice(
            request_id=rid,
            lag_ratio=float(lag_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
