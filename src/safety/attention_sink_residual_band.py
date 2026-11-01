"""AttentionSinkResidualBandAdvisor.

Advises sink_residual_ratio vs budgets.
Closes the vLLM/SGLang/TGI attention-sink residual monitors gap. Distinct from
`RopeScaleMismatchBandAdvisor` / `KvCacheQuantErrorBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AttentionSinkResidualBandAdvice:
    """AttentionSinkResidualBandAdvisor advice."""

    request_id: str
    sink_residual_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class AttentionSinkResidualBandAdvisor:
    """Advise sink_residual_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        sink_residual_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> AttentionSinkResidualBandAdvice:
        """Return sink_residual_ratio band.

        Args:
            request_id: Non-empty request id.
            sink_residual_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if sink_residual_ratio < 0:
            raise ValueError("sink_residual_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if sink_residual_ratio >= hard_limit:
            band = "breach"
        elif sink_residual_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return AttentionSinkResidualBandAdvice(
            request_id=rid,
            sink_residual_ratio=float(sink_residual_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
