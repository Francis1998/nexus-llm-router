"""Fair priority-lane queue for high / normal / bulk request classes."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class PriorityClass(StrEnum):
    """Request priority lane classes."""

    HIGH = "high"
    NORMAL = "normal"
    BULK = "bulk"


@dataclass(frozen=True, slots=True)
class QueuedRequest:
    """One enqueued request with its lane class."""

    request_id: str
    priority: PriorityClass


@dataclass(frozen=True, slots=True)
class LaneDepthSnapshot:
    """Per-lane and total depth view."""

    high: int
    normal: int
    bulk: int
    total: int


_DEFAULT_WEIGHTS: dict[PriorityClass, int] = {
    PriorityClass.HIGH: 4,
    PriorityClass.NORMAL: 2,
    PriorityClass.BULK: 1,
}


class RequestPriorityLane:
    """Weighted fair dequeue across high / normal / bulk request lanes.

    Closes the LiteLLM priority-queue / OpenRouter ranking gap for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 ingress: callers enqueue with a
    priority class and ``dequeue()`` serves a weighted fair schedule so bulk
    traffic cannot starve interactive work (and high cannot fully starve bulk).
    """

    def __init__(
        self,
        *,
        weights: Mapping[PriorityClass | str, int] | None = None,
    ) -> None:
        """Initialize empty per-lane queues.

        Args:
            weights: Positive integer weights per ``PriorityClass``. Defaults to
                ``{high: 4, normal: 2, bulk: 1}``. Missing classes fall back to
                the defaults; all resulting weights must be ``> 0``.

        Raises:
            ValueError: If any resolved weight is non-positive.
        """
        resolved = dict(_DEFAULT_WEIGHTS)
        if weights is not None:
            for key, value in weights.items():
                lane = PriorityClass(key) if not isinstance(key, PriorityClass) else key
                resolved[lane] = int(value)
        for lane, weight in resolved.items():
            if weight <= 0:
                raise ValueError(f"weight for {lane.value} must be > 0")
        self._weights = resolved
        self._queues: dict[PriorityClass, deque[str]] = {
            PriorityClass.HIGH: deque(),
            PriorityClass.NORMAL: deque(),
            PriorityClass.BULK: deque(),
        }
        self._cycle = self._build_cycle(resolved)
        self._cycle_index = 0
        self._lock = threading.Lock()

    @staticmethod
    def _build_cycle(weights: Mapping[PriorityClass, int]) -> tuple[PriorityClass, ...]:
        cycle: list[PriorityClass] = []
        # Stable lane order keeps fairness deterministic across processes.
        for lane in (PriorityClass.HIGH, PriorityClass.NORMAL, PriorityClass.BULK):
            cycle.extend([lane] * weights[lane])
        return tuple(cycle)

    @property
    def weights(self) -> dict[PriorityClass, int]:
        """Return a copy of the configured lane weights."""
        return dict(self._weights)

    def _normalize_priority(self, priority: PriorityClass | str) -> PriorityClass:
        if isinstance(priority, PriorityClass):
            return priority
        try:
            return PriorityClass(str(priority).strip().lower())
        except ValueError as exc:
            raise ValueError("priority must be one of: high, normal, bulk") from exc

    def enqueue(
        self, request_id: str, priority: PriorityClass | str = PriorityClass.NORMAL
    ) -> None:
        """Append ``request_id`` onto the matching priority lane.

        Args:
            request_id: Non-empty request identifier.
            priority: Lane class (``high`` / ``normal`` / ``bulk``).

        Raises:
            ValueError: If ``request_id`` is empty or ``priority`` is unknown.
        """
        if not request_id:
            raise ValueError("request_id must be non-empty")
        lane = self._normalize_priority(priority)
        with self._lock:
            self._queues[lane].append(request_id)

    def dequeue(self) -> QueuedRequest | None:
        """Pop the next request using weighted fair lane rotation.

        Empty lanes are skipped within the current cycle so a quiet high lane
        does not block normal/bulk work. Returns ``None`` when every lane is
        empty.
        """
        with self._lock:
            if not any(self._queues.values()):
                return None
            checked = 0
            while checked < len(self._cycle):
                lane = self._cycle[self._cycle_index % len(self._cycle)]
                self._cycle_index = (self._cycle_index + 1) % len(self._cycle)
                checked += 1
                queue = self._queues[lane]
                if queue:
                    return QueuedRequest(request_id=queue.popleft(), priority=lane)
            return None

    def depth(self, priority: PriorityClass | str | None = None) -> int:
        """Return depth for one lane, or total depth when ``priority`` is omitted."""
        with self._lock:
            if priority is None:
                return sum(len(queue) for queue in self._queues.values())
            lane = self._normalize_priority(priority)
            return len(self._queues[lane])

    def snapshot(self) -> LaneDepthSnapshot:
        """Return per-lane and total queue depths."""
        with self._lock:
            high = len(self._queues[PriorityClass.HIGH])
            normal = len(self._queues[PriorityClass.NORMAL])
            bulk = len(self._queues[PriorityClass.BULK])
            return LaneDepthSnapshot(
                high=high,
                normal=normal,
                bulk=bulk,
                total=high + normal + bulk,
            )

    def pending(self) -> list[QueuedRequest]:
        """Return queued requests in lane order (high, then normal, then bulk)."""
        with self._lock:
            items: list[QueuedRequest] = []
            for lane in (PriorityClass.HIGH, PriorityClass.NORMAL, PriorityClass.BULK):
                items.extend(
                    QueuedRequest(request_id=request_id, priority=lane)
                    for request_id in self._queues[lane]
                )
            return items

    def clear(self) -> None:
        """Drop every pending request and reset the fair-cycle cursor."""
        with self._lock:
            for queue in self._queues.values():
                queue.clear()
            self._cycle_index = 0
