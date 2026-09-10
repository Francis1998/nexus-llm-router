"""In-memory per-model latency SLA tracker (advisory p50/p95 windows)."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LatencySlaSnapshot:
    """Rolling latency window view for one model."""

    model: str
    samples: int
    p50_ms: float
    p95_ms: float
    sla_p95_ms: float
    breached: bool


class ModelLatencySlaTracker:
    """Track per-model p50/p95 latency windows and flag SLA breaches.

    Advisory and in-memory — closes the Helicone / Langfuse latency-dashboard
    gap for offline / air-gapped Nexus gateways serving GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 without requiring an external
    observability backend. Breaches never reject traffic; callers decide how
    to surface or act on ``breached`` snapshots.
    """

    def __init__(
        self,
        *,
        window_size: int = 100,
        sla_p95_ms: float = 2_000.0,
        per_model_sla_p95_ms: Mapping[str, float] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize an empty tracker.

        Args:
            window_size: Maximum samples retained per model (``>= 1``).
            sla_p95_ms: Default p95 SLA threshold in milliseconds (``> 0``).
            per_model_sla_p95_ms: Optional per-model p95 overrides.
            clock: Injectable clock (defaults to ``time.monotonic``) reserved
                for future time-based windows / tests.

        Raises:
            ValueError: If ``window_size`` or SLA thresholds are invalid.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if sla_p95_ms <= 0:
            raise ValueError("sla_p95_ms must be > 0")
        overrides: dict[str, float] = {}
        if per_model_sla_p95_ms is not None:
            for model, threshold in per_model_sla_p95_ms.items():
                if not model:
                    raise ValueError("per-model SLA keys must be non-empty")
                if threshold <= 0:
                    raise ValueError("per-model sla_p95_ms values must be > 0")
                overrides[model] = float(threshold)
        self._window_size = int(window_size)
        self._sla_p95_ms = float(sla_p95_ms)
        self._per_model_sla = overrides
        self._clock = clock or time.monotonic
        self._samples: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def window_size(self) -> int:
        """Return the configured per-model sample window size."""
        return self._window_size

    @property
    def sla_p95_ms(self) -> float:
        """Return the default p95 SLA threshold in milliseconds."""
        return self._sla_p95_ms

    def sla_for(self, model: str) -> float:
        """Return the effective p95 SLA for ``model``."""
        if not model:
            raise ValueError("model must be non-empty")
        return self._per_model_sla.get(model, self._sla_p95_ms)

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: float) -> float:
        if not sorted_values:
            return 0.0
        if len(sorted_values) == 1:
            return sorted_values[0]
        rank = percentile * (len(sorted_values) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = rank - lower
        return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

    def _snapshot_locked(self, model: str, values: deque[float]) -> LatencySlaSnapshot:
        ordered = sorted(values)
        p50 = self._percentile(ordered, 0.50)
        p95 = self._percentile(ordered, 0.95)
        threshold = self._per_model_sla.get(model, self._sla_p95_ms)
        return LatencySlaSnapshot(
            model=model,
            samples=len(ordered),
            p50_ms=p50,
            p95_ms=p95,
            sla_p95_ms=threshold,
            breached=p95 > threshold,
        )

    def record(self, model: str, latency_ms: float) -> LatencySlaSnapshot:
        """Record one observation and return the updated advisory snapshot.

        Args:
            model: Model identifier (for example ``gpt-5.5``).
            latency_ms: Observed end-to-end latency in milliseconds (``>= 0``).

        Returns:
            Updated ``LatencySlaSnapshot`` (``breached`` is advisory only).

        Raises:
            ValueError: If ``model`` is empty or ``latency_ms`` is negative.
        """
        if not model:
            raise ValueError("model must be non-empty")
        if latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        # Touch the clock so injectable clocks stay exercised in tests.
        _ = float(self._clock())
        with self._lock:
            window = self._samples.setdefault(model, deque(maxlen=self._window_size))
            window.append(float(latency_ms))
            return self._snapshot_locked(model, window)

    def snapshot(self, model: str) -> LatencySlaSnapshot | None:
        """Return the current snapshot for ``model``, or ``None`` if unknown."""
        if not model:
            raise ValueError("model must be non-empty")
        with self._lock:
            window = self._samples.get(model)
            if window is None or not window:
                return None
            return self._snapshot_locked(model, window)

    def breaches(self) -> list[LatencySlaSnapshot]:
        """Return snapshots for every model currently breaching its p95 SLA."""
        with self._lock:
            result: list[LatencySlaSnapshot] = []
            for model, window in self._samples.items():
                if not window:
                    continue
                snap = self._snapshot_locked(model, window)
                if snap.breached:
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
