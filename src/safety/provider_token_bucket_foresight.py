"""Provider token-bucket foresight advisor.

Advises whether remaining token-bucket capacity covers forecast demand.
Distinct from tenant rate limiters (hard caps) and ``CostForecastAdvisor``
(cost USD). Closes OpenRouter / LiteLLM / Portkey token-bucket foresight
gaps. Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderTokenBucketForesightAdvice:
    """Provider token-bucket foresight advice."""

    provider: str
    tokens_remaining: float
    forecast_demand: float
    ratio: float
    band: str


class ProviderTokenBucketForesightAdvisor:
    """Advise token-bucket remaining vs forecast demand."""

    def advise(
        self,
        *,
        provider: str,
        tokens_remaining: float,
        forecast_demand: float,
    ) -> ProviderTokenBucketForesightAdvice:
        """Return band for remaining tokens vs forecast demand.

        Args:
            provider: Non-empty provider name.
            tokens_remaining: Tokens left in bucket (``>= 0``).
            forecast_demand: Forecast token demand (``> 0``).

        Returns:
            ProviderTokenBucketForesightAdvice with
            ``ample`` / ``tight`` / ``exhausted``.
        """

        name = provider.strip()
        if not name:
            raise ValueError("provider must be non-empty")
        if tokens_remaining < 0:
            raise ValueError("tokens_remaining must be >= 0")
        if forecast_demand <= 0:
            raise ValueError("forecast_demand must be > 0")

        ratio = round(tokens_remaining / forecast_demand, 4)
        if ratio >= 1.2:
            band = "ample"
        elif ratio >= 1.0:
            band = "tight"
        else:
            band = "exhausted"
        return ProviderTokenBucketForesightAdvice(
            provider=name,
            tokens_remaining=float(tokens_remaining),
            forecast_demand=float(forecast_demand),
            ratio=float(ratio),
            band=band,
        )
