"""DecodePreemptionPingPongBandAdvisor.

Advises ping_pong_ratio vs budgets.
Closes the vLLM/SGLang/TGI decode preemption ping-pong monitors gap. Distinct from
`ContinuousBatchPreemptionChurnAdvisor` / `ContinuousBatchStarveBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecodePreemptionPingPongBandAdvice:
    """DecodePreemptionPingPongBandAdvisor advice."""

    request_id: str
    ping_pong_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class DecodePreemptionPingPongBandAdvisor:
    """Advise ping_pong_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        ping_pong_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> DecodePreemptionPingPongBandAdvice:
        """Return ping_pong_ratio band.

        Args:
            request_id: Non-empty request id.
            ping_pong_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if ping_pong_ratio < 0:
            raise ValueError("ping_pong_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if ping_pong_ratio >= hard_limit:
            band = "breach"
        elif ping_pong_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return DecodePreemptionPingPongBandAdvice(
            request_id=rid,
            ping_pong_ratio=float(ping_pong_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
