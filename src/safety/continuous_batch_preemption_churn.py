"""ContinuousBatchPreemptionChurnAdvisor.

Advises preemption_churn_rate vs budgets.
Closes the vLLM/SGLang/TGI continuous-batch preemption-churn monitors gap. Distinct from
`ContinuousBatchingTailLatencyBandAdvisor` / `PrefillQueueStarvationAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContinuousBatchPreemptionChurnAdvice:
    """ContinuousBatchPreemptionChurnAdvisor advice."""

    request_id: str
    preemption_churn_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class ContinuousBatchPreemptionChurnAdvisor:
    """Advise preemption_churn_rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        preemption_churn_rate: float,
        soft_limit: float = 0.25,
        hard_limit: float = 0.6,
    ) -> ContinuousBatchPreemptionChurnAdvice:
        """Return preemption_churn_rate band.

        Args:
            request_id: Non-empty request id.
            preemption_churn_rate: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if preemption_churn_rate < 0:
            raise ValueError("preemption_churn_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if preemption_churn_rate >= hard_limit:
            band = "breach"
        elif preemption_churn_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ContinuousBatchPreemptionChurnAdvice(
            request_id=rid,
            preemption_churn_rate=float(preemption_churn_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
