"""Provider warm-pool readiness advisor (cold / warming / hot).

Advises pool readiness from idle warm replicas vs target. Closes the
OpenRouter / LiteLLM / Portkey warm-pool gap. Distinct from
``ProviderHealthGuard`` (error rates) and ``RequestHedgingAdvisor`` (duplicate
hedges). Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WarmBand = Literal["cold", "warming", "hot"]


@dataclass(frozen=True, slots=True)
class ProviderWarmPoolAdvice:
    """Warm-pool readiness advice for one provider."""

    provider_id: str
    band: WarmBand
    warm_replicas: int
    target_replicas: int
    ready_ratio: float
    advisory: str


class ProviderWarmPoolAdvisor:
    """Advise cold/warming/hot bands from warm replica counts."""

    def __init__(self, *, hot_ratio: float = 1.0, warming_ratio: float = 0.5) -> None:
        if not 0.0 < warming_ratio <= 1.0:
            raise ValueError("warming_ratio must be in (0, 1]")
        if hot_ratio < warming_ratio:
            raise ValueError("hot_ratio must be >= warming_ratio")
        self._hot_ratio = float(hot_ratio)
        self._warming_ratio = float(warming_ratio)

    def advise(
        self,
        provider_id: str,
        *,
        warm_replicas: int,
        target_replicas: int,
    ) -> ProviderWarmPoolAdvice:
        """Return warm-pool advice for ``provider_id``."""

        if not provider_id:
            raise ValueError("provider_id must be non-empty")
        if warm_replicas < 0:
            raise ValueError("warm_replicas must be >= 0")
        if target_replicas <= 0:
            raise ValueError("target_replicas must be > 0")

        ratio = warm_replicas / float(target_replicas)
        if ratio >= self._hot_ratio:
            band: WarmBand = "hot"
        elif ratio >= self._warming_ratio:
            band = "warming"
        else:
            band = "cold"
        advisory = (
            f"{band}: warm_replicas={warm_replicas}/{target_replicas} "
            f"(ratio={ratio:.2f}) for provider_id={provider_id}"
        )
        return ProviderWarmPoolAdvice(
            provider_id=provider_id,
            band=band,
            warm_replicas=int(warm_replicas),
            target_replicas=int(target_replicas),
            ready_ratio=float(ratio),
            advisory=advisory,
        )
