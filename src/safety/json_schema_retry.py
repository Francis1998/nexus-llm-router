"""Per-request JSON schema repair retry budget (advisory / hard gate)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

RetryBand = Literal["ok", "advisory", "exhausted"]


@dataclass(frozen=True, slots=True)
class JsonSchemaRetrySnapshot:
    """Rolling JSON-schema repair retry view for one request_id."""

    request_id: str
    retries: int
    advisory_limit: int
    hard_limit: int
    band: RetryBand
    remaining_hard: int
    advisory: str
    blocked: bool


class JsonSchemaRetryBudgetExceededError(RuntimeError):
    """Raised by ``assert_within_budget`` when a hard gate blocks retries."""

    def __init__(self, snapshot: JsonSchemaRetrySnapshot) -> None:
        """Attach the exceeding snapshot to the exception.

        Args:
            snapshot: Retry-budget state at the moment of rejection.
        """
        self.snapshot = snapshot
        super().__init__(
            f"JSON schema retry budget exhausted for {snapshot.request_id}: "
            f"retries={snapshot.retries} hard_limit={snapshot.hard_limit}"
        )


class JsonSchemaRetryBudgetGuard:
    """Track JSON schema repair retries per request_id with advisory/hard budgets.

    Closes the LiteLLM / OpenRouter *structured-output repair retry budget*
    gap for offline Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2: callers record each repair attempt and learn whether
    they are still ``ok``, in the ``advisory`` band, or ``exhausted``.

    Distinct from ``OutputJsonSchemaGuard`` (post-check required-keys
    validation) and from ``RetryBudgetAwareFailoverStrategy`` (provider
    failover routing). Optional ``hard_gate`` raises on ``assert_within_budget``.
    """

    def __init__(
        self,
        *,
        advisory_limit: int = 2,
        hard_limit: int = 5,
        hard_gate: bool = False,
    ) -> None:
        """Initialize empty per-request retry counters.

        Args:
            advisory_limit: Retry count at/above which the band becomes
                ``advisory`` while still under the hard limit (``>= 1``).
            hard_limit: Retry count at/above which the band becomes
                ``exhausted`` (must be ``> advisory_limit``).
            hard_gate: When ``True``, ``assert_within_budget`` raises if
                exhausted.

        Raises:
            ValueError: If limits are invalid.
        """
        if advisory_limit < 1:
            raise ValueError("advisory_limit must be >= 1")
        if hard_limit <= advisory_limit:
            raise ValueError("hard_limit must be > advisory_limit")
        self._advisory_limit = int(advisory_limit)
        self._hard_limit = int(hard_limit)
        self._hard_gate = bool(hard_gate)
        self._retries: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def advisory_limit(self) -> int:
        """Return the configured advisory retry threshold."""
        return self._advisory_limit

    @property
    def hard_limit(self) -> int:
        """Return the configured hard retry ceiling."""
        return self._hard_limit

    @property
    def hard_gate(self) -> bool:
        """Return whether ``assert_within_budget`` raises when exhausted."""
        return self._hard_gate

    def _snapshot_locked(self, request_id: str, retries: int) -> JsonSchemaRetrySnapshot:
        if retries >= self._hard_limit:
            band: RetryBand = "exhausted"
            blocked = True
            advisory = (
                f"exhausted: retries={retries} >= hard_limit={self._hard_limit} "
                f"for request_id={request_id}"
            )
        elif retries >= self._advisory_limit:
            band = "advisory"
            blocked = False
            advisory = (
                f"advisory: retries={retries} >= advisory_limit="
                f"{self._advisory_limit} (hard_limit={self._hard_limit}) "
                f"for request_id={request_id}"
            )
        else:
            band = "ok"
            blocked = False
            advisory = (
                f"ok: retries={retries} < advisory_limit={self._advisory_limit} "
                f"for request_id={request_id}"
            )
        remaining = max(0, self._hard_limit - retries)
        return JsonSchemaRetrySnapshot(
            request_id=request_id,
            retries=int(retries),
            advisory_limit=int(self._advisory_limit),
            hard_limit=int(self._hard_limit),
            band=band,
            remaining_hard=int(remaining),
            advisory=advisory,
            blocked=blocked,
        )

    def record_retry(self, request_id: str) -> JsonSchemaRetrySnapshot:
        """Increment the repair-retry counter for ``request_id`` and snapshot.

        Args:
            request_id: Non-empty request identifier.

        Returns:
            Updated ``JsonSchemaRetrySnapshot``.

        Raises:
            ValueError: If ``request_id`` is empty.
        """
        if not request_id:
            raise ValueError("request_id must be non-empty")
        with self._lock:
            retries = self._retries.get(request_id, 0) + 1
            self._retries[request_id] = retries
            return self._snapshot_locked(request_id, retries)

    def snapshot(self, request_id: str) -> JsonSchemaRetrySnapshot | None:
        """Return the current snapshot for ``request_id``, or ``None`` if unknown."""
        if not request_id:
            raise ValueError("request_id must be non-empty")
        with self._lock:
            if request_id not in self._retries:
                return None
            return self._snapshot_locked(request_id, self._retries[request_id])

    def remaining(self, request_id: str) -> int:
        """Return remaining hard-budget retries for ``request_id`` (full if unknown)."""
        if not request_id:
            raise ValueError("request_id must be non-empty")
        with self._lock:
            retries = self._retries.get(request_id, 0)
            return max(0, self._hard_limit - retries)

    def allow_retry(self, request_id: str) -> bool:
        """Return whether another repair retry is under the hard limit.

        Unknown request ids are allowed (cold start). Exhausted ids are not.

        Args:
            request_id: Non-empty request identifier.

        Returns:
            ``True`` when ``retries < hard_limit`` (or the id is unknown).
        """
        if not request_id:
            raise ValueError("request_id must be non-empty")
        with self._lock:
            retries = self._retries.get(request_id, 0)
            return retries < self._hard_limit

    def assert_within_budget(self, request_id: str) -> JsonSchemaRetrySnapshot | None:
        """Hard-gate helper: raise when ``hard_gate`` and budget is exhausted.

        Args:
            request_id: Non-empty request identifier.

        Returns:
            Current snapshot, or ``None`` when the request id is unknown.

        Raises:
            JsonSchemaRetryBudgetExceededError: When ``hard_gate`` is enabled
                and the band is ``exhausted``.
            ValueError: If ``request_id`` is empty.
        """
        snap = self.snapshot(request_id)
        if self._hard_gate and snap is not None and snap.blocked:
            raise JsonSchemaRetryBudgetExceededError(snap)
        return snap

    def request_ids(self) -> list[str]:
        """Return known request ids in insertion order."""
        with self._lock:
            return list(self._retries.keys())

    def clear(self, request_id: str | None = None) -> None:
        """Clear one request counter, or every counter when omitted."""
        with self._lock:
            if request_id is None:
                self._retries.clear()
                return
            self._retries.pop(request_id, None)
