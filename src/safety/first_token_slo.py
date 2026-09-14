"""Advisory first-token (TTFT) latency SLO band advisor."""

from __future__ import annotations

import statistics
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

FirstTokenBand = Literal["within", "warn", "breach"]


@dataclass(frozen=True, slots=True)
class FirstTokenSloAdvice:
    """Structured TTFT / first-token SLO advisory for one observation."""

    model: str
    ttft_ms: float
    slo_ttft_ms: float
    warn_threshold_ms: float
    band: FirstTokenBand
    headroom_ms: float
    advisory: str
    samples: int = 1
    p50_ttft_ms: float | None = None


class FirstTokenLatencySloAdvisor:
    """Classify time-to-first-token against per-model TTFT SLO bands.

    Closes the Helicone / Langfuse *streaming TTFT* dashboard gap for offline
    Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2:
    operators need first-token (streaming start) SLO bands that are distinct
    from end-to-end completion latency.

    Distinct from ``ModelLatencySlaTracker`` (rolling end-to-end p50/p95) and
    from ``latency-slo-shed`` / ``latency-budget`` *routing strategies*. This
    advisor never rejects traffic — callers log or alert on ``band``.
    """

    def __init__(
        self,
        *,
        slo_ttft_ms: float = 500.0,
        warn_ratio: float = 0.8,
        window_size: int = 50,
        per_model_slo_ttft_ms: Mapping[str, float] | None = None,
    ) -> None:
        """Initialize TTFT SLO thresholds and an empty rolling window.

        Args:
            slo_ttft_ms: Default first-token SLO in milliseconds (``> 0``).
            warn_ratio: Fraction of the SLO in ``(0.0, 1.0)`` at/above which
                the band becomes ``warn`` while still under the hard SLO.
            window_size: Max TTFT samples retained per model (``>= 1``).
            per_model_slo_ttft_ms: Optional per-model TTFT SLO overrides.

        Raises:
            ValueError: If thresholds or overrides are invalid.
        """
        if slo_ttft_ms <= 0:
            raise ValueError("slo_ttft_ms must be > 0")
        if not 0.0 < warn_ratio < 1.0:
            raise ValueError("warn_ratio must be in (0.0, 1.0)")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        overrides: dict[str, float] = {}
        if per_model_slo_ttft_ms is not None:
            for model, threshold in per_model_slo_ttft_ms.items():
                if not model:
                    raise ValueError("per-model SLO keys must be non-empty")
                if threshold <= 0:
                    raise ValueError("per-model slo_ttft_ms values must be > 0")
                overrides[model] = float(threshold)
        self._slo_ttft_ms = float(slo_ttft_ms)
        self._warn_ratio = float(warn_ratio)
        self._window_size = int(window_size)
        self._per_model_slo = overrides
        self._samples: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def slo_ttft_ms(self) -> float:
        """Return the default TTFT SLO in milliseconds."""
        return self._slo_ttft_ms

    @property
    def warn_ratio(self) -> float:
        """Return the warn-band utilization fraction of the TTFT SLO."""
        return self._warn_ratio

    @property
    def window_size(self) -> int:
        """Return the per-model rolling sample window size."""
        return self._window_size

    def slo_for(self, model: str) -> float:
        """Return the effective TTFT SLO for ``model``."""
        if not model:
            raise ValueError("model must be non-empty")
        return self._per_model_slo.get(model, self._slo_ttft_ms)

    def _classify(
        self,
        model: str,
        ttft_ms: float,
        samples: int,
        p50: float | None,
    ) -> FirstTokenSloAdvice:
        slo = self.slo_for(model)
        warn_at = slo * self._warn_ratio
        headroom = slo - ttft_ms
        if ttft_ms > slo:
            band: FirstTokenBand = "breach"
            advisory = (
                f"breach: ttft_ms={ttft_ms:.1f} exceeds slo_ttft_ms={slo:.1f} "
                f"(headroom_ms={headroom:.1f})"
            )
        elif ttft_ms >= warn_at:
            band = "warn"
            advisory = (
                f"warn: ttft_ms={ttft_ms:.1f} >= warn_threshold_ms={warn_at:.1f} "
                f"(slo_ttft_ms={slo:.1f})"
            )
        else:
            band = "within"
            advisory = (
                f"within: ttft_ms={ttft_ms:.1f} under warn_threshold_ms={warn_at:.1f} "
                f"with headroom_ms={headroom:.1f}"
            )
        return FirstTokenSloAdvice(
            model=model,
            ttft_ms=float(ttft_ms),
            slo_ttft_ms=float(slo),
            warn_threshold_ms=float(warn_at),
            band=band,
            headroom_ms=float(headroom),
            advisory=advisory,
            samples=samples,
            p50_ttft_ms=p50,
        )

    def advise(self, *, model: str, ttft_ms: float) -> FirstTokenSloAdvice:
        """Return a one-shot TTFT band without mutating rolling state.

        Args:
            model: Model identifier (non-empty).
            ttft_ms: Observed time-to-first-token in milliseconds (``>= 0``).

        Returns:
            Immutable ``FirstTokenSloAdvice`` with band and headroom.

        Raises:
            ValueError: If ``model`` is empty or ``ttft_ms`` is negative.
        """
        if not model:
            raise ValueError("model must be non-empty")
        if ttft_ms < 0:
            raise ValueError("ttft_ms must be >= 0")
        return self._classify(model, float(ttft_ms), samples=1, p50=float(ttft_ms))

    def record(self, model: str, ttft_ms: float) -> FirstTokenSloAdvice:
        """Append a TTFT sample and return the rolling advisory snapshot.

        The band is evaluated against the newest sample (streaming start of
        the latest request). ``p50_ttft_ms`` reflects the rolling window.

        Args:
            model: Model identifier (non-empty).
            ttft_ms: Observed TTFT in milliseconds (``>= 0``).

        Returns:
            ``FirstTokenSloAdvice`` including rolling ``samples`` / ``p50``.

        Raises:
            ValueError: If ``model`` is empty or ``ttft_ms`` is negative.
        """
        if not model:
            raise ValueError("model must be non-empty")
        if ttft_ms < 0:
            raise ValueError("ttft_ms must be >= 0")
        with self._lock:
            window = self._samples.setdefault(model, deque(maxlen=self._window_size))
            window.append(float(ttft_ms))
            values = list(window)
            p50 = float(statistics.median(values))
            return self._classify(model, float(ttft_ms), samples=len(values), p50=p50)

    def snapshot(self, model: str) -> FirstTokenSloAdvice | None:
        """Return the latest rolling snapshot for ``model``, or ``None``."""
        if not model:
            raise ValueError("model must be non-empty")
        with self._lock:
            window = self._samples.get(model)
            if window is None or not window:
                return None
            values = list(window)
            p50 = float(statistics.median(values))
            return self._classify(model, float(values[-1]), samples=len(values), p50=p50)

    def breaches(self) -> list[FirstTokenSloAdvice]:
        """Return sorted snapshots for models whose latest TTFT breaches SLO."""
        with self._lock:
            result: list[FirstTokenSloAdvice] = []
            for model, window in self._samples.items():
                if not window:
                    continue
                values = list(window)
                p50 = float(statistics.median(values))
                snap = self._classify(model, float(values[-1]), samples=len(values), p50=p50)
                if snap.band == "breach":
                    result.append(snap)
            result.sort(key=lambda item: item.model)
            return result

    def models(self) -> list[str]:
        """Return known model names in insertion order."""
        with self._lock:
            return list(self._samples.keys())

    def clear(self, model: str | None = None) -> None:
        """Clear one model window, or every window when ``model`` is omitted."""
        with self._lock:
            if model is None:
                self._samples.clear()
                return
            self._samples.pop(model, None)
