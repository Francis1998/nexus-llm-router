"""Speculative-decode abort-rate advisor.

Advises speculative-decode abort-rate bands vs a budget. Closes the vLLM /
TensorRT-LLM / OpenRouter speculative-decode abort gap. Distinct from
``SpeculativeDecodeBudgetAdvisor`` (token budget) and
``OutputTokenForecastAdvisor`` (output forecast). Works with GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs network I/O.
"""


from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeDecodeAbortAdvice:
    """Speculative-decode abort advice."""

    request_id: str
    abort_rate: float
    budget_rate: float
    ratio: float
    band: str


class SpeculativeDecodeAbortAdvisor:
    """Advise speculative-decode abort rate vs budget bands."""

    def advise(
        self,
        *,
        request_id: str,
        abort_rate: float,
        budget_rate: float,
    ) -> SpeculativeDecodeAbortAdvice:
        """Return band for abort rate vs budget.

        Args:
            request_id: Non-empty request id.
            abort_rate: Observed abort rate in ``[0, 1]``.
            budget_rate: Abort budget rate in ``(0, 1]``.

        Returns:
            SpeculativeDecodeAbortAdvice with ``healthy`` / ``elevated`` / ``critical``.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if not 0.0 <= abort_rate <= 1.0:
            raise ValueError("abort_rate must be in [0, 1]")
        if not 0.0 < budget_rate <= 1.0:
            raise ValueError("budget_rate must be in (0, 1]")

        ratio = round(abort_rate / budget_rate, 4)
        if ratio <= 1.0:
            band = "healthy"
        elif ratio <= 1.5:
            band = "elevated"
        else:
            band = "critical"
        return SpeculativeDecodeAbortAdvice(
            request_id=rid,
            abort_rate=float(abort_rate),
            budget_rate=float(budget_rate),
            ratio=float(ratio),
            band=band,
        )
