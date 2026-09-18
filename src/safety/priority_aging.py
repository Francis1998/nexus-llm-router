"""Advisory request priority-aging boost bands (fresh / aging / stale)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AgingBand = Literal["fresh", "aging", "stale"]


@dataclass(frozen=True, slots=True)
class PriorityAgingAdvice:
    """Structured advisory for one wait-age → priority boost decision."""

    age_seconds: float
    band: AgingBand
    boost: int
    aging_after_seconds: float
    stale_after_seconds: float
    advisory: str
    request_id: str | None = None


class RequestPriorityAgingAdvisor:
    """Classify queue wait age into fresh / aging / stale priority boosts.

    Closes the Portkey / Helicone *queue wait-age fairness* gap for offline
    Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2:
    operators need wait-time → priority-boost bands so long-waiting requests
    can be promoted without replacing lane-class scheduling.

    Distinct from ``RequestPriorityLane`` (weighted fair high/normal/bulk
    dequeue). This advisor never mutates a queue — callers apply ``boost``.
    """

    def __init__(
        self,
        *,
        aging_after_seconds: float = 5.0,
        stale_after_seconds: float = 30.0,
        aging_boost: int = 1,
        stale_boost: int = 2,
    ) -> None:
        """Initialize aging thresholds and boost magnitudes.

        Args:
            aging_after_seconds: Age at/above which the band becomes ``aging``
                (``> 0``).
            stale_after_seconds: Age at/above which the band becomes ``stale``
                (must be ``> aging_after_seconds``).
            aging_boost: Non-negative integer boost for the aging band.
            stale_boost: Integer boost for the stale band (must be
                ``>= aging_boost``).

        Raises:
            ValueError: If thresholds or boosts are invalid.
        """
        if aging_after_seconds <= 0.0:
            raise ValueError("aging_after_seconds must be > 0")
        if stale_after_seconds <= aging_after_seconds:
            raise ValueError("stale_after_seconds must be > aging_after_seconds")
        if aging_boost < 0:
            raise ValueError("aging_boost must be >= 0")
        if stale_boost < aging_boost:
            raise ValueError("stale_boost must be >= aging_boost")
        self._aging_after_seconds = float(aging_after_seconds)
        self._stale_after_seconds = float(stale_after_seconds)
        self._aging_boost = int(aging_boost)
        self._stale_boost = int(stale_boost)

    @property
    def aging_after_seconds(self) -> float:
        """Return the configured aging threshold in seconds."""
        return self._aging_after_seconds

    @property
    def stale_after_seconds(self) -> float:
        """Return the configured stale threshold in seconds."""
        return self._stale_after_seconds

    @property
    def aging_boost(self) -> int:
        """Return the configured boost for the aging band."""
        return self._aging_boost

    @property
    def stale_boost(self) -> int:
        """Return the configured boost for the stale band."""
        return self._stale_boost

    def advise(
        self,
        *,
        age_seconds: float,
        request_id: str | None = None,
    ) -> PriorityAgingAdvice:
        """Return a fresh / aging / stale advisory for the wait age.

        Args:
            age_seconds: How long the request has waited (``>= 0``).
            request_id: Optional request id recorded on the advice for logging.

        Returns:
            Immutable ``PriorityAgingAdvice`` with band, boost, and advisory
            text. Never rejects and never mutates a queue.

        Raises:
            ValueError: If ``age_seconds`` is negative.
        """
        if age_seconds < 0.0:
            raise ValueError("age_seconds must be >= 0")

        if age_seconds >= self._stale_after_seconds:
            band: AgingBand = "stale"
            boost = self._stale_boost
            advisory = (
                f"stale: age_seconds={age_seconds} >= stale_after_seconds="
                f"{self._stale_after_seconds}; boost={boost}"
            )
        elif age_seconds >= self._aging_after_seconds:
            band = "aging"
            boost = self._aging_boost
            advisory = (
                f"aging: age_seconds={age_seconds} >= aging_after_seconds="
                f"{self._aging_after_seconds}; boost={boost}"
            )
        else:
            band = "fresh"
            boost = 0
            advisory = (
                f"fresh: age_seconds={age_seconds} < aging_after_seconds="
                f"{self._aging_after_seconds}; boost=0"
            )

        return PriorityAgingAdvice(
            age_seconds=float(age_seconds),
            band=band,
            boost=int(boost),
            aging_after_seconds=float(self._aging_after_seconds),
            stale_after_seconds=float(self._stale_after_seconds),
            advisory=advisory,
            request_id=request_id,
        )
