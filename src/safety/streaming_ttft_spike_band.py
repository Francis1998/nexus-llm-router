"""StreamingTtftSpikeBandAdvisor.

Advises spike_ratio vs budgets.
Closes the vLLM/SGLang/TGI streaming TTFT spike monitors gap. Distinct from
`PrefillTtftRatioAdvisor` / `FirstTokenLatencySloAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StreamingTtftSpikeBandAdvice:
    """StreamingTtftSpikeBandAdvisor advice."""

    request_id: str
    spike_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class StreamingTtftSpikeBandAdvisor:
    """Advise spike_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        spike_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> StreamingTtftSpikeBandAdvice:
        """Return spike_ratio band.

        Args:
            request_id: Non-empty request id.
            spike_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if spike_ratio < 0:
            raise ValueError("spike_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if spike_ratio >= hard_limit:
            band = "breach"
        elif spike_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return StreamingTtftSpikeBandAdvice(
            request_id=rid,
            spike_ratio=float(spike_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
