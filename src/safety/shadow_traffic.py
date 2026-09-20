"""Shadow traffic mirror guard (off / mirror / full).

Controls what fraction of production traffic may be mirrored as shadow
requests to a candidate provider. Closes the Portkey / Helicone / OpenRouter
*shadow traffic* gap. Distinct from ``ProviderCanaryRolloutGuard`` (canary %)
and ``StickyProviderAffinity`` (session stickiness). Works with GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ShadowBand = Literal["off", "mirror", "full"]


@dataclass(frozen=True, slots=True)
class ShadowTrafficDecision:
    """Shadow-mirror decision for one request."""

    request_id: str
    band: ShadowBand
    mirror_pct: float
    should_mirror: bool
    advisory: str


class ShadowTrafficMirrorGuard:
    """Decide whether to mirror a request as shadow traffic."""

    def __init__(self, *, mirror_pct: float = 5.0) -> None:
        if not 0.0 <= mirror_pct <= 100.0:
            raise ValueError("mirror_pct must be in [0, 100]")
        self._mirror_pct = float(mirror_pct)

    @property
    def mirror_pct(self) -> float:
        return self._mirror_pct

    def decide(self, request_id: str, *, sample_unit: float) -> ShadowTrafficDecision:
        """Return shadow decision for ``request_id``.

        Args:
            request_id: Non-empty request id.
            sample_unit: Deterministic sample in ``[0, 1)`` (e.g. hash fraction).

        Returns:
            ShadowTrafficDecision with band off/mirror/full.
        """
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if not 0.0 <= sample_unit < 1.0:
            raise ValueError("sample_unit must be in [0, 1)")

        if self._mirror_pct <= 0.0:
            band: ShadowBand = "off"
            should = False
            advisory = f"off: mirror_pct=0 for request_id={request_id}"
        elif self._mirror_pct >= 100.0:
            band = "full"
            should = True
            advisory = f"full: mirror_pct=100 for request_id={request_id}"
        else:
            threshold = self._mirror_pct / 100.0
            should = sample_unit < threshold
            band = "mirror"
            advisory = (
                f"mirror: sample_unit={sample_unit:.4f} "
                f"{'<' if should else '>='} threshold={threshold:.4f} "
                f"for request_id={request_id}"
            )
        return ShadowTrafficDecision(
            request_id=request_id,
            band=band,
            mirror_pct=self._mirror_pct,
            should_mirror=should,
            advisory=advisory,
        )
