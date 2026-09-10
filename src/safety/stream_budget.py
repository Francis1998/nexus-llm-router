"""Hard mid-stream token/cost budget gate for streaming completions."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StreamBudgetSnapshot:
    """Current token/cost consumption against configured hard budgets."""

    tokens_used: int
    cost_usd: float
    max_tokens: int | None
    max_cost_usd: float | None
    exhausted: bool
    reason: str | None


class StreamingTokenBudgetExceededError(RuntimeError):
    """Raised when a stream chunk would exceed the hard token or cost budget."""

    def __init__(self, snapshot: StreamBudgetSnapshot) -> None:
        """Attach the exhausting snapshot to the exception.

        Args:
            snapshot: Budget state at the moment of exhaustion.
        """
        self.snapshot = snapshot
        detail = snapshot.reason or "streaming token/cost budget exhausted"
        super().__init__(detail)


class StreamingTokenBudgetGate:
    """Hard-stop mid-stream when token or cost budget is exhausted.

    Closes the LiteLLM / OpenRouter soft max-token / soft spend-limit gap for
    GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 streaming responses:
    callers observe each chunk and either continue, get an exhausted snapshot,
    or raise ``StreamingTokenBudgetExceededError`` for a hard cut-off.
    """

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        """Initialize hard budgets for one stream (or a reusable gate).

        Args:
            max_tokens: Maximum cumulative completion tokens. ``None`` disables
                the token ceiling.
            max_cost_usd: Maximum cumulative USD cost. ``None`` disables the
                cost ceiling.

        Raises:
            ValueError: If both budgets are unset, or either is non-positive.
        """
        if max_tokens is None and max_cost_usd is None:
            raise ValueError("at least one of max_tokens or max_cost_usd must be set")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be > 0 when set")
        if max_cost_usd is not None and max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be > 0 when set")
        self._max_tokens = max_tokens
        self._max_cost_usd = max_cost_usd
        self._tokens_used = 0
        self._cost_usd = 0.0
        self._exhausted = False
        self._reason: str | None = None
        self._lock = threading.Lock()

    @property
    def max_tokens(self) -> int | None:
        """Return the configured hard token budget, or ``None`` when disabled."""
        return self._max_tokens

    @property
    def max_cost_usd(self) -> float | None:
        """Return the configured hard cost budget, or ``None`` when disabled."""
        return self._max_cost_usd

    def _snapshot_locked(self) -> StreamBudgetSnapshot:
        return StreamBudgetSnapshot(
            tokens_used=self._tokens_used,
            cost_usd=self._cost_usd,
            max_tokens=self._max_tokens,
            max_cost_usd=self._max_cost_usd,
            exhausted=self._exhausted,
            reason=self._reason,
        )

    def _evaluate_exhaustion_locked(self) -> None:
        if self._exhausted:
            return
        if self._max_tokens is not None and self._tokens_used >= self._max_tokens:
            self._exhausted = True
            self._reason = (
                f"token budget exhausted: used={self._tokens_used} max={self._max_tokens}"
            )
            return
        if self._max_cost_usd is not None and self._cost_usd >= self._max_cost_usd:
            self._exhausted = True
            self._reason = (
                f"cost budget exhausted: used={self._cost_usd:.6f} max={self._max_cost_usd:.6f}"
            )

    def snapshot(self) -> StreamBudgetSnapshot:
        """Return the current budget snapshot."""
        with self._lock:
            return self._snapshot_locked()

    def remaining_tokens(self) -> int | None:
        """Return remaining token budget, or ``None`` when no token ceiling."""
        with self._lock:
            if self._max_tokens is None:
                return None
            return max(0, self._max_tokens - self._tokens_used)

    def remaining_cost_usd(self) -> float | None:
        """Return remaining cost budget, or ``None`` when no cost ceiling."""
        with self._lock:
            if self._max_cost_usd is None:
                return None
            return max(0.0, self._max_cost_usd - self._cost_usd)

    def allow_continue(self) -> bool:
        """Return ``True`` while the stream may emit additional chunks."""
        with self._lock:
            return not self._exhausted

    def observe_chunk(
        self,
        *,
        tokens: int = 0,
        cost_usd: float = 0.0,
        hard_stop: bool = False,
    ) -> StreamBudgetSnapshot:
        """Account for one streamed chunk and optionally hard-stop on breach.

        Args:
            tokens: Completion tokens attributed to this chunk (``>= 0``).
            cost_usd: USD cost attributed to this chunk (``>= 0``).
            hard_stop: When ``True``, raise if the budget is exhausted after
                applying this chunk (or was already exhausted).

        Returns:
            Updated ``StreamBudgetSnapshot``.

        Raises:
            ValueError: If ``tokens`` or ``cost_usd`` is negative.
            StreamingTokenBudgetExceededError: When ``hard_stop`` is ``True``
                and the budget is exhausted.
        """
        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        if cost_usd < 0:
            raise ValueError("cost_usd must be >= 0")
        with self._lock:
            if not self._exhausted:
                self._tokens_used += int(tokens)
                self._cost_usd += float(cost_usd)
                self._evaluate_exhaustion_locked()
            snapshot = self._snapshot_locked()
        if hard_stop and snapshot.exhausted:
            raise StreamingTokenBudgetExceededError(snapshot)
        return snapshot

    def assert_within_budget(self) -> None:
        """Raise when the gate is already exhausted.

        Raises:
            StreamingTokenBudgetExceededError: If a prior chunk exhausted the
                budget.
        """
        with self._lock:
            snapshot = self._snapshot_locked()
        if snapshot.exhausted:
            raise StreamingTokenBudgetExceededError(snapshot)

    def reset(self) -> None:
        """Clear usage counters so the gate can be reused for another stream."""
        with self._lock:
            self._tokens_used = 0
            self._cost_usd = 0.0
            self._exhausted = False
            self._reason = None
