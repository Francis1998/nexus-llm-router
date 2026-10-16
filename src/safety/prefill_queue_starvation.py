"""PrefillQueueStarvationAdvisor.

Advises starvation_rate vs budgets.
Closes the vLLM/SGLang/TensorRT-LLM prefill queue-starvation monitors gap. Distinct from
`PrefillDecodeSkewAdvisor` / `PrefillTtftRatioAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefillQueueStarvationAdvice:
    """PrefillQueueStarvationAdvisor advice."""

    request_id: str
    starvation_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class PrefillQueueStarvationAdvisor:
    """Advise starvation_rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        starvation_rate: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> PrefillQueueStarvationAdvice:
        """Return starvation_rate band.

        Args:
            request_id: Non-empty request id.
            starvation_rate: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if starvation_rate < 0:
            raise ValueError("starvation_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if starvation_rate >= hard_limit:
            band = "breach"
        elif starvation_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return PrefillQueueStarvationAdvice(
            request_id=rid,
            starvation_rate=float(starvation_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
