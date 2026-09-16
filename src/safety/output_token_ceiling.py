"""Advisory output-token ceiling bands (ok / near / over)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CeilingBand = Literal["ok", "near", "over"]


@dataclass(frozen=True, slots=True)
class OutputTokenCeilingAdvice:
    """Structured advisory for one requested max_tokens vs a ceiling."""

    requested_max_tokens: int
    ceiling: int
    band: CeilingBand
    utilization: float
    headroom_tokens: int
    near_ratio: float
    advisory: str
    model: str | None = None


class OutputTokenCeilingGuard:
    """Classify requested max_tokens against an output-token ceiling.

    Closes the LiteLLM / OpenRouter / Portkey *max_tokens vs soft ceiling*
    advisory gap for offline Nexus gateways serving GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2: operators need ``ok`` /
    ``near`` / ``over`` bands before dispatch when a request asks for more
    completion tokens than a policy ceiling allows.

    Distinct from ``StreamingTokenBudgetGate`` (hard mid-stream cut-off) and
    ``ContextWindowFitAdvisor`` (prompt tokens vs model context window).
    This guard never kills requests itself — callers clamp, warn, or proceed.
    """

    def __init__(self, *, near_ratio: float = 0.85) -> None:
        """Initialize near-band utilization threshold.

        Args:
            near_ratio: Utilization fraction in ``(0.0, 1.0)`` at/above which
                the band becomes ``near`` while still under the hard ceiling.

        Raises:
            ValueError: If ``near_ratio`` is not strictly between 0 and 1.
        """
        if not 0.0 < near_ratio < 1.0:
            raise ValueError("near_ratio must be in (0.0, 1.0)")
        self._near_ratio = float(near_ratio)

    @property
    def near_ratio(self) -> float:
        """Return the configured near-band utilization threshold."""
        return self._near_ratio

    def advise(
        self,
        *,
        requested_max_tokens: int,
        ceiling: int,
        model: str | None = None,
    ) -> OutputTokenCeilingAdvice:
        """Return an ok / near / over advisory for the requested max tokens.

        Args:
            requested_max_tokens: Caller-requested completion budget (``>= 0``).
            ceiling: Policy / account output-token ceiling (``>= 1``).
            model: Optional model id recorded on the advice for logging.

        Returns:
            Immutable ``OutputTokenCeilingAdvice`` with band, utilization,
            headroom (negative when over), and advisory text. Never rejects.

        Raises:
            ValueError: If token counts are invalid.
        """
        if requested_max_tokens < 0:
            raise ValueError("requested_max_tokens must be >= 0")
        if ceiling < 1:
            raise ValueError("ceiling must be >= 1")

        utilization = requested_max_tokens / float(ceiling)
        headroom = ceiling - requested_max_tokens

        if requested_max_tokens > ceiling:
            band: CeilingBand = "over"
            advisory = (
                f"over: requested_max_tokens={requested_max_tokens} exceeds "
                f"ceiling={ceiling} (utilization={utilization:.3f}); "
                f"clamp or lower max_tokens"
            )
        elif utilization >= self._near_ratio:
            band = "near"
            advisory = (
                f"near: utilization={utilization:.3f} >= near_ratio="
                f"{self._near_ratio:.3f}; headroom_tokens={headroom} "
                f"(ceiling={ceiling})"
            )
        else:
            band = "ok"
            advisory = (
                f"ok: utilization={utilization:.3f} with "
                f"headroom_tokens={headroom} under ceiling={ceiling}"
            )

        return OutputTokenCeilingAdvice(
            requested_max_tokens=int(requested_max_tokens),
            ceiling=int(ceiling),
            band=band,
            utilization=utilization,
            headroom_tokens=int(headroom),
            near_ratio=float(self._near_ratio),
            advisory=advisory,
            model=model,
        )
