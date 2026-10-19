"""ContinuousBatchingTailLatencyBandAdvisor.

Advises p99_tail_ms vs budgets.
Closes the vLLM/SGLang/TGI continuous-batching tail-latency band monitors gap. Distinct from
`DecodeBatchPriorityInversionAdvisor` / `GrammarConstrainedDecodeRejectAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContinuousBatchingTailLatencyBandAdvice:
    """ContinuousBatchingTailLatencyBandAdvisor advice."""

    request_id: str
    p99_tail_ms: float
    soft_limit: float
    hard_limit: float
    band: str


class ContinuousBatchingTailLatencyBandAdvisor:
    """Advise p99_tail_ms bands."""

    def advise(
        self,
        *,
        request_id: str,
        p99_tail_ms: float,
        soft_limit: float = 250.0,
        hard_limit: float = 800.0,
    ) -> ContinuousBatchingTailLatencyBandAdvice:
        """Return p99_tail_ms band.

        Args:
            request_id: Non-empty request id.
            p99_tail_ms: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if p99_tail_ms < 0:
            raise ValueError("p99_tail_ms must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if p99_tail_ms >= hard_limit:
            band = "breach"
        elif p99_tail_ms >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ContinuousBatchingTailLatencyBandAdvice(
            request_id=rid,
            p99_tail_ms=float(p99_tail_ms),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
