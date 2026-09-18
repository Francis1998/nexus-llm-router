"""Advisory model temperature clamp bands (ok / high / extreme)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TemperatureBand = Literal["ok", "high", "extreme"]


@dataclass(frozen=True, slots=True)
class TemperatureClampAdvice:
    """Structured advisory for one requested temperature vs a max_allowed."""

    temperature: float
    max_allowed: float
    band: TemperatureBand
    high_ratio: float
    suggested_clamp: float
    advisory: str
    model: str | None = None


class ModelTemperatureClampAdvisor:
    """Classify requested temperature against a policy max_allowed clamp.

    Closes the LiteLLM / OpenRouter / Portkey *temperature vs soft max*
    advisory gap for offline Nexus gateways serving GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2: operators need ``ok`` /
    ``high`` / ``extreme`` bands before dispatch when a request asks for a
    temperature near or above a policy ceiling.

    This advisor never rejects traffic itself — callers clamp, warn, or
    proceed from ``band`` / ``suggested_clamp``.
    """

    def __init__(self, *, high_ratio: float = 0.85) -> None:
        """Initialize high-band utilization threshold.

        Args:
            high_ratio: Fraction of ``max_allowed`` in ``(0.0, 1.0)`` at/above
                which the band becomes ``high`` while still at or under the
                hard ceiling.

        Raises:
            ValueError: If ``high_ratio`` is not strictly between 0 and 1.
        """
        if not 0.0 < high_ratio < 1.0:
            raise ValueError("high_ratio must be in (0.0, 1.0)")
        self._high_ratio = float(high_ratio)

    @property
    def high_ratio(self) -> float:
        """Return the configured high-band utilization threshold."""
        return self._high_ratio

    def advise(
        self,
        *,
        temperature: float,
        max_allowed: float,
        model: str | None = None,
    ) -> TemperatureClampAdvice:
        """Return an ok / high / extreme advisory for the requested temperature.

        Args:
            temperature: Caller-requested sampling temperature (``>= 0``).
            max_allowed: Policy / account temperature ceiling (``> 0``).
            model: Optional model id recorded on the advice for logging.

        Returns:
            Immutable ``TemperatureClampAdvice`` with band, suggested clamp
            (``min(temperature, max_allowed)``), and advisory text. Never
            rejects.

        Raises:
            ValueError: If temperature or max_allowed are invalid.
        """
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if max_allowed <= 0.0:
            raise ValueError("max_allowed must be > 0")

        utilization = temperature / float(max_allowed)
        suggested = min(float(temperature), float(max_allowed))

        if temperature > max_allowed:
            band: TemperatureBand = "extreme"
            advisory = (
                f"extreme: temperature={temperature} exceeds "
                f"max_allowed={max_allowed} (utilization={utilization:.3f}); "
                f"clamp to suggested_clamp={suggested}"
            )
        elif utilization >= self._high_ratio:
            band = "high"
            advisory = (
                f"high: utilization={utilization:.3f} >= high_ratio="
                f"{self._high_ratio:.3f}; temperature={temperature} under "
                f"max_allowed={max_allowed}"
            )
        else:
            band = "ok"
            advisory = (
                f"ok: utilization={utilization:.3f} with "
                f"temperature={temperature} under max_allowed={max_allowed}"
            )

        return TemperatureClampAdvice(
            temperature=float(temperature),
            max_allowed=float(max_allowed),
            band=band,
            high_ratio=float(self._high_ratio),
            suggested_clamp=float(suggested),
            advisory=advisory,
            model=model,
        )
