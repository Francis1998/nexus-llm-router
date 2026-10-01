"""Tool-arg size foresight advisor.

Advises tool-argument byte size vs a budget with foresight bands.
Closes the OpenRouter / LiteLLM / Portkey tool-arg size foresight
gap. Distinct from ``ParallelToolCallArityAdvisor`` and
``MultimodalPayloadSizeGate``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolArgSizeForesightAdvice:
    """Tool-arg size foresight advice."""

    request_id: str
    arg_bytes: int
    budget_bytes: int
    utilization: float
    soft_limit: float
    hard_limit: float
    band: str


class ToolArgSizeForesightAdvisor:
    """Advise tool-arg size foresight bands."""

    def advise(
        self,
        *,
        request_id: str,
        arg_bytes: int,
        budget_bytes: int,
        soft_limit: float = 0.7,
        hard_limit: float = 1.0,
    ) -> ToolArgSizeForesightAdvice:
        """Return utilization band.

        Args:
            request_id: Non-empty request id.
            arg_bytes: Forecast tool-arg bytes (``>= 0``).
            budget_bytes: Byte budget (``> 0``).
            soft_limit: Soft utilization (``> 0``).
            hard_limit: Hard utilization (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if arg_bytes < 0:
            raise ValueError("arg_bytes must be >= 0")
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be > 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        utilization = round(arg_bytes / budget_bytes, 4)
        if utilization >= hard_limit:
            band = "breach"
        elif utilization >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ToolArgSizeForesightAdvice(
            request_id=rid,
            arg_bytes=int(arg_bytes),
            budget_bytes=int(budget_bytes),
            utilization=float(utilization),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
