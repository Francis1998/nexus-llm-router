"""Advisory context-window fit bands from estimated tokens vs model limit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FitBand = Literal["fits", "tight", "overflow"]


@dataclass(frozen=True, slots=True)
class ContextWindowFitAdvice:
    """Structured fit advisory for one prospective prompt vs a model limit."""

    estimated_tokens: int
    model_context_limit: int
    band: FitBand
    utilization: float
    headroom_tokens: int
    advisory: str
    model: str | None = None


class ContextWindowFitAdvisor:
    """Classify estimated prompt size against a model's context window.

    Closes the LiteLLM / OpenRouter pre-dispatch context-fit advisory gap for
    Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2:
    given ``estimated_tokens`` and ``model_context_limit``, return a fit band
    (``fits`` / ``tight`` / ``overflow``) plus a human-readable advisory.

    Distinct from ``StreamingTokenBudgetGate`` (hard mid-stream cut-off) and
    ``RequestCostForecastAdvisor`` (USD estimate). This advisor never mutates
    state and never rejects traffic — callers decide whether to truncate,
    pick a longer-context model, or proceed.
    """

    def __init__(self, *, tight_ratio: float = 0.85) -> None:
        """Initialize advisor thresholds.

        Args:
            tight_ratio: Utilization fraction in ``(0.0, 1.0)`` at/above which
                the band becomes ``tight`` while still under the hard limit.

        Raises:
            ValueError: If ``tight_ratio`` is not strictly between 0 and 1.
        """
        if not 0.0 < tight_ratio < 1.0:
            raise ValueError("tight_ratio must be in (0.0, 1.0)")
        self._tight_ratio = float(tight_ratio)

    @property
    def tight_ratio(self) -> float:
        """Return the configured tight-band utilization threshold."""
        return self._tight_ratio

    def advise(
        self,
        *,
        estimated_tokens: int,
        model_context_limit: int,
        model: str | None = None,
    ) -> ContextWindowFitAdvice:
        """Return a fit band and advisory for the estimated token load.

        Args:
            estimated_tokens: Counted or estimated prompt (+ planned reply)
                tokens (``>= 0``).
            model_context_limit: Model context window in tokens (``>= 1``).
            model: Optional model id recorded on the advice for logging.

        Returns:
            Immutable ``ContextWindowFitAdvice`` with band, utilization,
            headroom (negative when overflowing), and advisory text.

        Raises:
            ValueError: If token counts are invalid.
        """
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must be >= 0")
        if model_context_limit < 1:
            raise ValueError("model_context_limit must be >= 1")

        utilization = estimated_tokens / float(model_context_limit)
        headroom = model_context_limit - estimated_tokens

        if estimated_tokens >= model_context_limit:
            band: FitBand = "overflow"
            advisory = (
                f"overflow: estimated_tokens={estimated_tokens} meets/exceeds "
                f"model_context_limit={model_context_limit} "
                f"(utilization={utilization:.3f}); truncate or pick a longer window"
            )
        elif utilization >= self._tight_ratio:
            band = "tight"
            advisory = (
                f"tight: utilization={utilization:.3f} >= tight_ratio="
                f"{self._tight_ratio:.3f}; headroom_tokens={headroom}"
            )
        else:
            band = "fits"
            advisory = (
                f"fits: utilization={utilization:.3f} with "
                f"headroom_tokens={headroom} under limit={model_context_limit}"
            )

        return ContextWindowFitAdvice(
            estimated_tokens=int(estimated_tokens),
            model_context_limit=int(model_context_limit),
            band=band,
            utilization=utilization,
            headroom_tokens=int(headroom),
            advisory=advisory,
            model=model,
        )
