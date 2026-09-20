"""Request hedging advisor (hedge / hold / skip).

Advises when to fire a hedged duplicate request based on observed wait age vs
SLO. Closes the LiteLLM / Portkey / OpenRouter *request hedging* gap.
Distinct from ``RequestPriorityAgingAdvisor`` (priority boost bands) and
``FirstTokenLatencySloAdvisor`` (TTFT bands). Works with GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HedgeBand = Literal["hold", "hedge", "skip"]


@dataclass(frozen=True, slots=True)
class RequestHedgingAdvice:
    """Hedging advice for one in-flight request."""

    request_id: str
    band: HedgeBand
    wait_ms: float
    slo_ms: float
    hedge: bool
    advisory: str


class RequestHedgingAdvisor:
    """Advise hedged duplicate requests from wait age vs SLO."""

    def __init__(self, *, slo_ms: float = 800.0, hedge_ratio: float = 0.75) -> None:
        if slo_ms <= 0:
            raise ValueError("slo_ms must be > 0")
        if not 0.0 < hedge_ratio < 1.0:
            raise ValueError("hedge_ratio must be in (0, 1)")
        self._slo_ms = float(slo_ms)
        self._hedge_ratio = float(hedge_ratio)

    def advise(self, request_id: str, *, wait_ms: float) -> RequestHedgingAdvice:
        """Return hedging advice for ``request_id``.

        Args:
            request_id: Non-empty request id.
            wait_ms: Observed wait/TTFB age in milliseconds (``>= 0``).

        Returns:
            RequestHedgingAdvice with band hold/hedge/skip.
        """
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if wait_ms < 0:
            raise ValueError("wait_ms must be >= 0")

        threshold = self._slo_ms * self._hedge_ratio
        if wait_ms >= self._slo_ms:
            band: HedgeBand = "skip"
            hedge = False
            advisory = (
                f"skip: wait_ms={wait_ms:.1f} >= slo_ms={self._slo_ms} for request_id={request_id}"
            )
        elif wait_ms >= threshold:
            band = "hedge"
            hedge = True
            advisory = (
                f"hedge: wait_ms={wait_ms:.1f} >= threshold={threshold:.1f} "
                f"for request_id={request_id}"
            )
        else:
            band = "hold"
            hedge = False
            advisory = (
                f"hold: wait_ms={wait_ms:.1f} < threshold={threshold:.1f} "
                f"for request_id={request_id}"
            )
        return RequestHedgingAdvice(
            request_id=request_id,
            band=band,
            wait_ms=float(wait_ms),
            slo_ms=self._slo_ms,
            hedge=hedge,
            advisory=advisory,
        )
