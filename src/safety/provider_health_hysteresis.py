"""Provider health hysteresis advisor.

Advises enter/exit hysteresis bands for provider unhealthy transitions to
reduce flap. Closes the OpenRouter / LiteLLM / Portkey health-flap gap.
Distinct from ``ProviderHealthScoreboard`` (instant health) and
``ProviderErrorBudgetShed`` (error budget). Works with GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs network I/O.
"""


from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderHealthHysteresisAdvice:
    """Provider health hysteresis advice."""

    provider_id: str
    error_rate: float
    enter_threshold: float
    exit_threshold: float
    currently_unhealthy: bool
    band: str


class ProviderHealthHysteresisAdvisor:
    """Advise provider health with enter/exit hysteresis bands."""

    def advise(
        self,
        *,
        provider_id: str,
        error_rate: float,
        enter_threshold: float,
        exit_threshold: float,
        currently_unhealthy: bool,
    ) -> ProviderHealthHysteresisAdvice:
        """Return hysteresis band for provider health.

        Args:
            provider_id: Non-empty provider id.
            error_rate: Observed error rate in ``[0, 1]``.
            enter_threshold: Enter-unhealthy threshold in ``(0, 1]``.
            exit_threshold: Exit-unhealthy threshold in ``(0, enter_threshold]``.
            currently_unhealthy: Whether provider is currently marked unhealthy.

        Returns:
            ProviderHealthHysteresisAdvice with ``healthy`` / ``hold`` / ``unhealthy``.
        """

        pid = provider_id.strip()
        if not pid:
            raise ValueError("provider_id must be non-empty")
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be in [0, 1]")
        if not 0.0 < enter_threshold <= 1.0:
            raise ValueError("enter_threshold must be in (0, 1]")
        if not 0.0 < exit_threshold <= enter_threshold:
            raise ValueError("exit_threshold must be in (0, enter_threshold]")

        if currently_unhealthy:
            band = "healthy" if error_rate <= exit_threshold else "hold"
            if error_rate > enter_threshold:
                band = "unhealthy"
        else:
            if error_rate >= enter_threshold:
                band = "unhealthy"
            elif error_rate >= exit_threshold:
                band = "hold"
            else:
                band = "healthy"

        return ProviderHealthHysteresisAdvice(
            provider_id=pid,
            error_rate=float(error_rate),
            enter_threshold=float(enter_threshold),
            exit_threshold=float(exit_threshold),
            currently_unhealthy=bool(currently_unhealthy),
            band=band,
        )
