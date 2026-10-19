"""SpeculativeTokenVerifyLagAdvisor.

Advises verify_lag_ms vs budgets.
Closes the vLLM/SGLang/TGI speculative-token verify-lag monitors gap. Distinct from
`SpeculativeDecodeMismatchFloorAdvisor` / `SpeculativeDecodeRejectionStormAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeTokenVerifyLagAdvice:
    """SpeculativeTokenVerifyLagAdvisor advice."""

    request_id: str
    verify_lag_ms: float
    soft_limit: float
    hard_limit: float
    band: str


class SpeculativeTokenVerifyLagAdvisor:
    """Advise verify_lag_ms bands."""

    def advise(
        self,
        *,
        request_id: str,
        verify_lag_ms: float,
        soft_limit: float = 40.0,
        hard_limit: float = 120.0,
    ) -> SpeculativeTokenVerifyLagAdvice:
        """Return verify_lag_ms band.

        Args:
            request_id: Non-empty request id.
            verify_lag_ms: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if verify_lag_ms < 0:
            raise ValueError("verify_lag_ms must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if verify_lag_ms >= hard_limit:
            band = "breach"
        elif verify_lag_ms >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return SpeculativeTokenVerifyLagAdvice(
            request_id=rid,
            verify_lag_ms=float(verify_lag_ms),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
