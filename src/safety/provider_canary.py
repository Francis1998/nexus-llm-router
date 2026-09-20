"""Provider canary rollout guard (off / canary / promote).

Controls what fraction of traffic may route to a canary provider during
rollout. Closes the OpenRouter / Portkey / LiteLLM *canary rollout* gap.
Distinct from ``ShadowTrafficMirrorGuard`` (shadow mirrors) and
``ProviderHealthScoreboard`` (health bands). Works with GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CanaryBand = Literal["off", "canary", "promote"]


@dataclass(frozen=True, slots=True)
class ProviderCanaryDecision:
    """Canary routing decision for one request."""

    request_id: str
    provider: str
    band: CanaryBand
    canary_pct: float
    use_canary: bool
    advisory: str


class ProviderCanaryRolloutGuard:
    """Decide whether to route a request to a canary provider."""

    def __init__(self, *, canary_pct: float = 5.0, provider: str = "canary") -> None:
        if not 0.0 <= canary_pct <= 100.0:
            raise ValueError("canary_pct must be in [0, 100]")
        if not provider.strip():
            raise ValueError("provider must be non-empty")
        self._canary_pct = float(canary_pct)
        self._provider = provider.strip()

    @property
    def canary_pct(self) -> float:
        return self._canary_pct

    @property
    def provider(self) -> str:
        return self._provider

    def decide(self, request_id: str, *, sample_unit: float) -> ProviderCanaryDecision:
        """Return canary decision for ``request_id``.

        Args:
            request_id: Non-empty request id.
            sample_unit: Deterministic sample in ``[0, 1)``.

        Returns:
            ProviderCanaryDecision with band off/canary/promote.
        """
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if not 0.0 <= sample_unit < 1.0:
            raise ValueError("sample_unit must be in [0, 1)")

        if self._canary_pct <= 0.0:
            band: CanaryBand = "off"
            use = False
            advisory = f"off: canary_pct=0 provider={self._provider}"
        elif self._canary_pct >= 100.0:
            band = "promote"
            use = True
            advisory = f"promote: canary_pct=100 provider={self._provider}"
        else:
            threshold = self._canary_pct / 100.0
            use = sample_unit < threshold
            band = "canary"
            advisory = (
                f"canary: sample_unit={sample_unit:.4f} "
                f"{'<' if use else '>='} threshold={threshold:.4f} "
                f"provider={self._provider}"
            )
        return ProviderCanaryDecision(
            request_id=request_id,
            provider=self._provider,
            band=band,
            canary_pct=self._canary_pct,
            use_canary=use,
            advisory=advisory,
        )
