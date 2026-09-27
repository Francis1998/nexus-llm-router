"""Output token forecast advisor.

Advises forecast bands from estimated completion tokens vs a budget.
Closes the OpenRouter / LiteLLM / Portkey output-token foresight gap.
Distinct from ``OutputTokenCeilingGuard`` and ``RequestCostForecastAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs
network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutputTokenForecastAdvice:
    """Output token forecast advice."""

    request_id: str
    estimated_tokens: int
    budget_tokens: int
    ratio: float
    band: str


class OutputTokenForecastAdvisor:
    """Advise output-token forecast vs budget bands."""

    def advise(
        self,
        *,
        request_id: str,
        estimated_tokens: int,
        budget_tokens: int,
    ) -> OutputTokenForecastAdvice:
        """Return band for estimated tokens vs budget.

        Args:
            request_id: Non-empty request id.
            estimated_tokens: Forecast completion tokens (``>= 0``).
            budget_tokens: Allowed completion tokens (``> 0``).

        Returns:
            OutputTokenForecastAdvice with ``fit`` / ``tight`` / ``over``.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must be >= 0")
        if budget_tokens <= 0:
            raise ValueError("budget_tokens must be > 0")

        ratio = round(estimated_tokens / float(budget_tokens), 4)
        if ratio >= 1.0:
            band = "over"
        elif ratio >= 0.8:
            band = "tight"
        else:
            band = "fit"
        return OutputTokenForecastAdvice(
            request_id=rid,
            estimated_tokens=int(estimated_tokens),
            budget_tokens=int(budget_tokens),
            ratio=float(ratio),
            band=band,
        )
