"""AdapterWeightDriftAdvisor.

Advises weight_drift_score vs budgets.
Closes the vLLM/TGI LoRA adapter-weight drift monitors gap. Distinct from
`LoraAdapterSwapCostAdvisor` / `EmbeddingDimMismatchAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdapterWeightDriftAdvice:
    """AdapterWeightDriftAdvisor advice."""

    request_id: str
    weight_drift_score: float
    soft_limit: float
    hard_limit: float
    band: str


class AdapterWeightDriftAdvisor:
    """Advise weight_drift_score bands."""

    def advise(
        self,
        *,
        request_id: str,
        weight_drift_score: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> AdapterWeightDriftAdvice:
        """Return weight_drift_score band.

        Args:
            request_id: Non-empty request id.
            weight_drift_score: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if weight_drift_score < 0:
            raise ValueError("weight_drift_score must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if weight_drift_score >= hard_limit:
            band = "breach"
        elif weight_drift_score >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return AdapterWeightDriftAdvice(
            request_id=rid,
            weight_drift_score=float(weight_drift_score),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
