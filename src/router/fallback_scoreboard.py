"""Provider health scoreboard for fallback ordering."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderHealthSnapshot:
    """Rolling health view for one provider."""

    provider: str
    attempts: float
    successes: float
    success_rate: float
    error_rate: float
    avg_latency_ms: float
    score: float


@dataclass
class _ProviderStats:
    """Mutable EWMA / decay-aware counters for one provider."""

    successes: float = 0.0
    attempts: float = 0.0
    latency_sum_ms: float = 0.0
    updated_at: float = 0.0


class ProviderFallbackScoreboard:
    """Track per-provider success / latency / error rates for fallback ranking.

    Closes the LiteLLM / OpenRouter provider-health scoring gap for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2: callers record outcomes and ask
    ``rank()`` for a recommended fallback order (healthiest first).

    An injectable ``clock`` keeps tests deterministic. Optional exponential
    decay (half-life in seconds) down-weights stale observations before each
    update / read.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        decay_half_life_seconds: float | None = None,
        latency_ref_ms: float = 1_000.0,
    ) -> None:
        """Initialize an empty scoreboard.

        Args:
            clock: Monotonic clock used for decay timestamps (defaults to
                ``time.monotonic``).
            decay_half_life_seconds: When set, observation weight halves every
                this many seconds. ``None`` disables decay.
            latency_ref_ms: Latency scale used when converting average latency
                into the health score denominator.
        """
        if decay_half_life_seconds is not None and decay_half_life_seconds <= 0:
            raise ValueError("decay_half_life_seconds must be > 0 when set")
        if latency_ref_ms <= 0:
            raise ValueError("latency_ref_ms must be > 0")
        self._clock = clock or time.monotonic
        self._decay_half_life_seconds = decay_half_life_seconds
        self._latency_ref_ms = float(latency_ref_ms)
        self._stats: dict[str, _ProviderStats] = {}
        self._lock = threading.Lock()

    @property
    def decay_half_life_seconds(self) -> float | None:
        """Return the configured decay half-life, or ``None`` when disabled."""
        return self._decay_half_life_seconds

    @property
    def latency_ref_ms(self) -> float:
        """Return the latency reference used in the health score."""
        return self._latency_ref_ms

    def _decay_factor(self, stats: _ProviderStats, now: float) -> float:
        if self._decay_half_life_seconds is None or stats.attempts <= 0:
            return 1.0
        age = max(0.0, now - stats.updated_at)
        if age <= 0:
            return 1.0
        return math.pow(0.5, age / self._decay_half_life_seconds)

    def _apply_decay_locked(self, stats: _ProviderStats, now: float) -> None:
        factor = self._decay_factor(stats, now)
        if factor == 1.0:
            return
        stats.successes *= factor
        stats.attempts *= factor
        stats.latency_sum_ms *= factor
        stats.updated_at = now

    def record_outcome(self, provider: str, success: bool, latency_ms: float) -> None:
        """Record one provider attempt outcome.

        Args:
            provider: Provider name (for example ``openai`` / ``anthropic``).
            success: Whether the attempt succeeded.
            latency_ms: Observed end-to-end latency in milliseconds.

        Raises:
            ValueError: If ``provider`` is empty or ``latency_ms`` is negative.
        """
        if not provider:
            raise ValueError("provider must be non-empty")
        if latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        now = float(self._clock())
        with self._lock:
            stats = self._stats.setdefault(provider, _ProviderStats(updated_at=now))
            self._apply_decay_locked(stats, now)
            stats.attempts += 1.0
            if success:
                stats.successes += 1.0
            stats.latency_sum_ms += float(latency_ms)
            stats.updated_at = now

    def _score_locked(self, stats: _ProviderStats, now: float) -> float:
        self._apply_decay_locked(stats, now)
        if stats.attempts <= 0:
            return 0.0
        success_rate = stats.successes / stats.attempts
        avg_latency = stats.latency_sum_ms / stats.attempts
        # Higher success and lower latency → higher score.
        return success_rate / (1.0 + (avg_latency / self._latency_ref_ms))

    def score(self, provider: str) -> float:
        """Return the current health score for ``provider`` (``0.0`` if unknown)."""
        if not provider:
            raise ValueError("provider must be non-empty")
        now = float(self._clock())
        with self._lock:
            stats = self._stats.get(provider)
            if stats is None:
                return 0.0
            return self._score_locked(stats, now)

    def snapshot(self, provider: str) -> ProviderHealthSnapshot | None:
        """Return a health snapshot for ``provider``, or ``None`` if unknown."""
        if not provider:
            raise ValueError("provider must be non-empty")
        now = float(self._clock())
        with self._lock:
            stats = self._stats.get(provider)
            if stats is None:
                return None
            score = self._score_locked(stats, now)
            attempts = stats.attempts
            successes = stats.successes
            success_rate = 0.0 if attempts <= 0 else successes / attempts
            avg_latency = 0.0 if attempts <= 0 else stats.latency_sum_ms / attempts
            return ProviderHealthSnapshot(
                provider=provider,
                attempts=attempts,
                successes=successes,
                success_rate=success_rate,
                error_rate=1.0 - success_rate,
                avg_latency_ms=avg_latency,
                score=score,
            )

    def rank(self, providers: Sequence[str] | Iterable[str] | None = None) -> list[str]:
        """Return providers ordered by health score (best first).

        Args:
            providers: Optional candidate list. When omitted, ranks every
                provider that has at least one recorded outcome. Unknown
                candidates (no observations) sort after known ones, preserving
                input order among ties / unknowns.

        Returns:
            Ordered provider names suitable as a fallback preference list.
        """
        now = float(self._clock())
        with self._lock:
            candidates = list(self._stats.keys()) if providers is None else list(providers)
            scored: list[tuple[float, int, str]] = []
            for index, provider in enumerate(candidates):
                stats = self._stats.get(provider)
                score = float("-inf") if stats is None else self._score_locked(stats, now)
                scored.append((score, index, provider))
            scored.sort(key=lambda item: (-item[0], item[1]))
            return [provider for _, _, provider in scored]

    def providers(self) -> list[str]:
        """Return known provider names in insertion order."""
        with self._lock:
            return list(self._stats.keys())
