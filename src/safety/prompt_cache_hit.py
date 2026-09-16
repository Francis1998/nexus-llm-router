"""Advisory prompt-cache hit-rate bands (cold / warm / hot)."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Literal

HitRateBand = Literal["cold", "warm", "hot"]


@dataclass(frozen=True, slots=True)
class PromptCacheHitAdvice:
    """Structured advisory for one prompt-cache hit-rate observation."""

    cached_tokens: int
    prompt_tokens: int
    hit_rate: float
    band: HitRateBand
    warm_min_hit_rate: float
    hot_min_hit_rate: float
    samples: int
    rolling_hit_rate: float | None
    advisory: str
    model: str | None = None


class PromptCacheHitRateAdvisor:
    """Classify prompt-cache hit rates into cold / warm / hot bands.

    Closes the OpenRouter / LiteLLM / Helicone *prompt-cache hit-rate*
    observability gap for offline Nexus gateways serving GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2: operators need advisory bands
    when ``cached_tokens / prompt_tokens`` is cold, warm, or hot.

    Distinct from ``RequestFingerprintDeduper`` (identical-request content-hash
    window) and ``SemanticCacheStrategy`` (near-duplicate response routing).
    This advisor never rejects traffic and never stores response bodies —
    callers log or alert on ``band``.
    """

    def __init__(
        self,
        *,
        warm_min_hit_rate: float = 0.25,
        hot_min_hit_rate: float = 0.75,
        window_size: int = 50,
    ) -> None:
        """Initialize hit-rate thresholds and an empty rolling window.

        Args:
            warm_min_hit_rate: Hit-rate floor for ``warm`` in ``(0.0, 1.0)``.
                Below this band is ``cold``.
            hot_min_hit_rate: Hit-rate floor for ``hot`` in ``(warm, 1.0]``.
            window_size: Max hit-rate samples retained (``>= 1``).

        Raises:
            ValueError: If thresholds or window_size are invalid.
        """
        if not 0.0 < warm_min_hit_rate < 1.0:
            raise ValueError("warm_min_hit_rate must be in (0.0, 1.0)")
        if not 0.0 < hot_min_hit_rate <= 1.0:
            raise ValueError("hot_min_hit_rate must be in (0.0, 1.0]")
        if hot_min_hit_rate <= warm_min_hit_rate:
            raise ValueError("hot_min_hit_rate must be > warm_min_hit_rate")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._warm_min = float(warm_min_hit_rate)
        self._hot_min = float(hot_min_hit_rate)
        self._window_size = int(window_size)
        self._samples: deque[float] = deque(maxlen=self._window_size)
        self._lock = threading.Lock()

    @property
    def warm_min_hit_rate(self) -> float:
        """Return the hit-rate floor for the warm band."""
        return self._warm_min

    @property
    def hot_min_hit_rate(self) -> float:
        """Return the hit-rate floor for the hot band."""
        return self._hot_min

    @property
    def window_size(self) -> int:
        """Return the rolling sample window size."""
        return self._window_size

    def _classify(
        self,
        *,
        cached_tokens: int,
        prompt_tokens: int,
        hit_rate: float,
        samples: int,
        rolling_hit_rate: float | None,
        model: str | None,
    ) -> PromptCacheHitAdvice:
        if hit_rate >= self._hot_min:
            band: HitRateBand = "hot"
            advisory = (
                f"hot: hit_rate={hit_rate:.3f} >= hot_min={self._hot_min:.3f} "
                f"(cached_tokens={cached_tokens}, prompt_tokens={prompt_tokens})"
            )
        elif hit_rate >= self._warm_min:
            band = "warm"
            advisory = (
                f"warm: hit_rate={hit_rate:.3f} >= warm_min={self._warm_min:.3f} "
                f"(hot_min={self._hot_min:.3f}; cached_tokens={cached_tokens}, "
                f"prompt_tokens={prompt_tokens})"
            )
        else:
            band = "cold"
            advisory = (
                f"cold: hit_rate={hit_rate:.3f} < warm_min={self._warm_min:.3f} "
                f"(cached_tokens={cached_tokens}, prompt_tokens={prompt_tokens})"
            )
        return PromptCacheHitAdvice(
            cached_tokens=int(cached_tokens),
            prompt_tokens=int(prompt_tokens),
            hit_rate=float(hit_rate),
            band=band,
            warm_min_hit_rate=float(self._warm_min),
            hot_min_hit_rate=float(self._hot_min),
            samples=int(samples),
            rolling_hit_rate=rolling_hit_rate,
            advisory=advisory,
            model=model,
        )

    @staticmethod
    def _hit_rate(cached_tokens: int, prompt_tokens: int) -> float:
        if prompt_tokens == 0:
            return 0.0
        return cached_tokens / float(prompt_tokens)

    def record(
        self,
        *,
        cached_tokens: int,
        prompt_tokens: int,
        model: str | None = None,
    ) -> PromptCacheHitAdvice:
        """Record a cache observation and return hit-rate band advice.

        Args:
            cached_tokens: Tokens served from the prompt cache (``>= 0``).
            prompt_tokens: Total prompt tokens for the call (``>= 0``).
                ``cached_tokens`` must not exceed ``prompt_tokens``.
            model: Optional model id recorded on the advice for logging.

        Returns:
            Immutable ``PromptCacheHitAdvice`` with band, hit_rate, and
            rolling window stats. Never rejects traffic.

        Raises:
            ValueError: If token counts are invalid.
        """
        if cached_tokens < 0:
            raise ValueError("cached_tokens must be >= 0")
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")
        if cached_tokens > prompt_tokens:
            raise ValueError("cached_tokens must be <= prompt_tokens")

        hit_rate = self._hit_rate(cached_tokens, prompt_tokens)
        with self._lock:
            self._samples.append(hit_rate)
            samples = len(self._samples)
            rolling = sum(self._samples) / float(samples) if samples else None
        return self._classify(
            cached_tokens=cached_tokens,
            prompt_tokens=prompt_tokens,
            hit_rate=hit_rate,
            samples=samples,
            rolling_hit_rate=rolling,
            model=model,
        )

    def snapshot(self) -> dict[str, float | int | None]:
        """Return rolling window stats without mutating state."""
        with self._lock:
            samples = len(self._samples)
            rolling = sum(self._samples) / float(samples) if samples else None
        return {
            "samples": samples,
            "rolling_hit_rate": rolling,
            "warm_min_hit_rate": self._warm_min,
            "hot_min_hit_rate": self._hot_min,
        }

    def clear(self) -> None:
        """Drop all rolling hit-rate samples."""
        with self._lock:
            self._samples.clear()
