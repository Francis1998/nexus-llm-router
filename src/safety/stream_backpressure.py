"""Advisory streaming inter-chunk backpressure / stall bands."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Literal

BackpressureBand = Literal["ok", "stall", "backpressure"]


@dataclass(frozen=True, slots=True)
class StreamBackpressureAdvice:
    """Structured inter-chunk gap advisory for one streaming observation."""

    stream_id: str
    gap_ms: float
    stall_threshold_ms: float
    backpressure_threshold_ms: float
    band: BackpressureBand
    samples: int
    p50_gap_ms: float | None
    advisory: str


class StreamingBackpressureAdvisor:
    """Classify inter-chunk gaps into ok / stall / backpressure bands.

    Closes the Helicone / LiteLLM *streaming backpressure* observability gap
    for offline Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2: operators need advisory bands when the gap between
    successive SSE chunks exceeds configured thresholds.

    Distinct from ``StreamingTokenBudgetGate`` (hard mid-stream token/cost
    cut-off) and ``FirstTokenLatencySloAdvisor`` (TTFT / first-token SLO only).
    This advisor never rejects traffic and never truncates streams — callers
    log or alert on ``band``.
    """

    def __init__(
        self,
        *,
        stall_threshold_ms: float = 250.0,
        backpressure_threshold_ms: float = 1_000.0,
        window_size: int = 50,
    ) -> None:
        """Initialize gap thresholds and an empty rolling window.

        Args:
            stall_threshold_ms: Gap at/above which the band becomes ``stall``
                while still under backpressure (``> 0``).
            backpressure_threshold_ms: Gap above which the band becomes
                ``backpressure`` (``> stall_threshold_ms``).
            window_size: Max gap samples retained per stream (``>= 1``).

        Raises:
            ValueError: If thresholds or window_size are invalid.
        """
        if stall_threshold_ms <= 0:
            raise ValueError("stall_threshold_ms must be > 0")
        if backpressure_threshold_ms <= stall_threshold_ms:
            raise ValueError("backpressure_threshold_ms must be > stall_threshold_ms")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._stall_threshold_ms = float(stall_threshold_ms)
        self._backpressure_threshold_ms = float(backpressure_threshold_ms)
        self._window_size = int(window_size)
        self._samples: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def stall_threshold_ms(self) -> float:
        """Return the stall-band gap threshold in milliseconds."""
        return self._stall_threshold_ms

    @property
    def backpressure_threshold_ms(self) -> float:
        """Return the backpressure-band gap threshold in milliseconds."""
        return self._backpressure_threshold_ms

    @property
    def window_size(self) -> int:
        """Return the per-stream rolling sample window size."""
        return self._window_size

    def _classify(
        self,
        stream_id: str,
        gap_ms: float,
        samples: int,
        p50: float | None,
    ) -> StreamBackpressureAdvice:
        if gap_ms > self._backpressure_threshold_ms:
            band: BackpressureBand = "backpressure"
            advisory = (
                f"backpressure: gap_ms={gap_ms:.1f} > "
                f"backpressure_threshold_ms={self._backpressure_threshold_ms:.1f}"
            )
        elif gap_ms >= self._stall_threshold_ms:
            band = "stall"
            advisory = (
                f"stall: gap_ms={gap_ms:.1f} >= "
                f"stall_threshold_ms={self._stall_threshold_ms:.1f} "
                f"(backpressure_threshold_ms={self._backpressure_threshold_ms:.1f})"
            )
        else:
            band = "ok"
            advisory = (
                f"ok: gap_ms={gap_ms:.1f} under stall_threshold_ms={self._stall_threshold_ms:.1f}"
            )
        return StreamBackpressureAdvice(
            stream_id=stream_id,
            gap_ms=float(gap_ms),
            stall_threshold_ms=float(self._stall_threshold_ms),
            backpressure_threshold_ms=float(self._backpressure_threshold_ms),
            band=band,
            samples=samples,
            p50_gap_ms=p50,
            advisory=advisory,
        )

    def advise(self, *, stream_id: str, gap_ms: float) -> StreamBackpressureAdvice:
        """Return a one-shot gap band without mutating rolling state.

        Args:
            stream_id: Stream / request identifier (non-empty).
            gap_ms: Observed inter-chunk gap in milliseconds (``>= 0``).

        Returns:
            Immutable ``StreamBackpressureAdvice`` with band and thresholds.

        Raises:
            ValueError: If ``stream_id`` is empty or ``gap_ms`` is negative.
        """
        if not stream_id:
            raise ValueError("stream_id must be non-empty")
        if gap_ms < 0:
            raise ValueError("gap_ms must be >= 0")
        return self._classify(stream_id, float(gap_ms), samples=1, p50=float(gap_ms))

    def record(self, stream_id: str, gap_ms: float) -> StreamBackpressureAdvice:
        """Append a gap sample and return the rolling advisory snapshot.

        The band is evaluated against the newest gap. ``p50_gap_ms`` reflects
        the rolling window median.

        Args:
            stream_id: Stream / request identifier (non-empty).
            gap_ms: Observed inter-chunk gap in milliseconds (``>= 0``).

        Returns:
            ``StreamBackpressureAdvice`` including rolling ``samples`` / ``p50``.

        Raises:
            ValueError: If ``stream_id`` is empty or ``gap_ms`` is negative.
        """
        if not stream_id:
            raise ValueError("stream_id must be non-empty")
        if gap_ms < 0:
            raise ValueError("gap_ms must be >= 0")
        with self._lock:
            window = self._samples.setdefault(stream_id, deque(maxlen=self._window_size))
            window.append(float(gap_ms))
            values = list(window)
            p50 = float(sorted(values)[len(values) // 2])
            return self._classify(stream_id, float(gap_ms), samples=len(values), p50=p50)

    def snapshot(self, stream_id: str) -> StreamBackpressureAdvice | None:
        """Return the latest rolling snapshot for ``stream_id``, or ``None``."""
        if not stream_id:
            raise ValueError("stream_id must be non-empty")
        with self._lock:
            window = self._samples.get(stream_id)
            if window is None or not window:
                return None
            values = list(window)
            p50 = float(sorted(values)[len(values) // 2])
            return self._classify(stream_id, float(values[-1]), samples=len(values), p50=p50)

    def backpressured(self) -> list[StreamBackpressureAdvice]:
        """Return sorted snapshots whose latest gap is in ``backpressure``."""
        with self._lock:
            result: list[StreamBackpressureAdvice] = []
            for stream_id, window in self._samples.items():
                if not window:
                    continue
                values = list(window)
                p50 = float(sorted(values)[len(values) // 2])
                snap = self._classify(stream_id, float(values[-1]), samples=len(values), p50=p50)
                if snap.band == "backpressure":
                    result.append(snap)
            result.sort(key=lambda item: item.stream_id)
            return result

    def streams(self) -> list[str]:
        """Return known stream ids in insertion order."""
        with self._lock:
            return list(self._samples.keys())

    def clear(self, stream_id: str | None = None) -> None:
        """Clear one stream window, or every window when ``stream_id`` is omitted."""
        with self._lock:
            if stream_id is None:
                self._samples.clear()
                return
            self._samples.pop(stream_id, None)
