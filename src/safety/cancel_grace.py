"""Streaming cancel-grace window guard (open / grace / closed)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

GraceBand = Literal["open", "grace", "closed"]


@dataclass(frozen=True, slots=True)
class CancelGraceSnapshot:
    """Cancel-grace window view for one stream/request."""

    request_id: str
    band: GraceBand
    grace_ms: float
    elapsed_ms: float
    allow_final_chunk: bool
    advisory: str


class StreamingCancelGraceGuard:
    """Track post-cancel grace windows for streaming responses.

    Closes the Portkey / LiteLLM / OpenRouter *cancel grace* gap where clients
    abort mid-stream but gateways still need a short window to emit a final
    chunk or usage trailer. Distinct from ``StreamingBackpressureAdvisor``
    (inter-chunk gaps) and ``FirstTokenLatencySloAdvisor`` (TTFT). Works with
    GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs network
    I/O.
    """

    def __init__(
        self,
        *,
        grace_ms: float = 250.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if grace_ms <= 0:
            raise ValueError("grace_ms must be > 0")
        self._grace_ms = float(grace_ms)
        self._clock = clock or time.monotonic
        self._cancel_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def cancel(self, request_id: str) -> CancelGraceSnapshot:
        """Mark ``request_id`` as cancelled and open the grace window."""
        if not request_id:
            raise ValueError("request_id must be non-empty")
        with self._lock:
            self._cancel_at[request_id] = float(self._clock())
            return self._snap_locked(request_id)

    def snapshot(self, request_id: str) -> CancelGraceSnapshot | None:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        with self._lock:
            if request_id not in self._cancel_at:
                return None
            return self._snap_locked(request_id)

    def allow_final_chunk(self, request_id: str) -> bool:
        """Return whether a final chunk is still allowed under grace."""
        snap = self.snapshot(request_id)
        if snap is None:
            return True
        return snap.allow_final_chunk

    def clear(self, request_id: str | None = None) -> None:
        with self._lock:
            if request_id is None:
                self._cancel_at.clear()
                return
            self._cancel_at.pop(request_id, None)

    def _snap_locked(self, request_id: str) -> CancelGraceSnapshot:
        started = self._cancel_at[request_id]
        elapsed_ms = max(0.0, (float(self._clock()) - started) * 1000.0)
        if elapsed_ms <= 0.0:
            band: GraceBand = "open"
            allow = True
            advisory = f"open: cancel just observed for request_id={request_id}"
        elif elapsed_ms <= self._grace_ms:
            band = "grace"
            allow = True
            advisory = (
                f"grace: elapsed_ms={elapsed_ms:.1f} <= grace_ms={self._grace_ms} "
                f"for request_id={request_id}"
            )
        else:
            band = "closed"
            allow = False
            advisory = (
                f"closed: elapsed_ms={elapsed_ms:.1f} > grace_ms={self._grace_ms} "
                f"for request_id={request_id}"
            )
        return CancelGraceSnapshot(
            request_id=request_id,
            band=band,
            grace_ms=self._grace_ms,
            elapsed_ms=round(elapsed_ms, 3),
            allow_final_chunk=allow,
            advisory=advisory,
        )
