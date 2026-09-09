"""Tests for provider fallback health scoreboard."""

from __future__ import annotations

import pytest

from router.fallback_scoreboard import ProviderFallbackScoreboard, ProviderHealthSnapshot


class _Clock:
    """Injectable monotonic clock for deterministic decay tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_rank_prefers_higher_success_lower_latency() -> None:
    """Healthy low-latency providers rank ahead of failing / slow ones."""
    board = ProviderFallbackScoreboard(latency_ref_ms=1000.0)
    # openai: always succeeds, fast
    board.record_outcome("openai", True, 100.0)
    board.record_outcome("openai", True, 120.0)
    # anthropic: mixed success, medium latency
    board.record_outcome("anthropic", True, 400.0)
    board.record_outcome("anthropic", False, 450.0)
    # google: failures dominate
    board.record_outcome("google", False, 200.0)
    board.record_outcome("google", False, 220.0)
    ordered = board.rank()
    assert ordered[0] == "openai"
    assert ordered[-1] == "google"
    assert board.score("openai") > board.score("anthropic") > board.score("google")


def test_rank_with_candidate_list_places_unknowns_last() -> None:
    """Unknown candidates keep relative order after known scored providers."""
    board = ProviderFallbackScoreboard()
    board.record_outcome("openai", True, 80.0)
    board.record_outcome("moonshot", False, 500.0)
    ordered = board.rank(["moonshot", "unknown-a", "openai", "unknown-b"])
    assert ordered[0] == "openai"
    assert ordered[1] == "moonshot"
    assert ordered[2:] == ["unknown-a", "unknown-b"]


def test_snapshot_reports_success_error_and_latency() -> None:
    """snapshot exposes success/error rates and average latency."""
    board = ProviderFallbackScoreboard()
    board.record_outcome("anthropic", True, 200.0)
    board.record_outcome("anthropic", False, 400.0)
    snap = board.snapshot("anthropic")
    assert isinstance(snap, ProviderHealthSnapshot)
    assert snap.attempts == 2.0
    assert snap.successes == 1.0
    assert snap.success_rate == pytest.approx(0.5)
    assert snap.error_rate == pytest.approx(0.5)
    assert snap.avg_latency_ms == pytest.approx(300.0)
    assert board.snapshot("missing") is None


def test_decay_downweights_stale_failures() -> None:
    """With decay enabled, old failures lose influence after half-life elapses."""
    clock = _Clock(start=0.0)
    board = ProviderFallbackScoreboard(clock=clock, decay_half_life_seconds=10.0)
    board.record_outcome("openai", False, 900.0)
    board.record_outcome("openai", False, 900.0)
    clock.now = 40.0  # four half-lives later
    board.record_outcome("openai", True, 100.0)
    snap = board.snapshot("openai")
    assert snap is not None
    assert snap.success_rate > 0.8
    assert board.rank() == ["openai"]


def test_record_outcome_rejects_invalid_inputs() -> None:
    """Empty provider names and negative latency are rejected."""
    board = ProviderFallbackScoreboard()
    with pytest.raises(ValueError, match="provider"):
        board.record_outcome("", True, 10.0)
    with pytest.raises(ValueError, match="latency"):
        board.record_outcome("openai", True, -1.0)


def test_decay_half_life_must_be_positive() -> None:
    """Non-positive decay half-life is rejected at construction."""
    with pytest.raises(ValueError, match="decay_half_life_seconds"):
        ProviderFallbackScoreboard(decay_half_life_seconds=0.0)
