"""Tests for ModelLatencySlaTracker advisory p50/p95 windows."""

from __future__ import annotations

import pytest

from safety.latency_sla import LatencySlaSnapshot, ModelLatencySlaTracker


class _Clock:
    """Injectable monotonic clock."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_record_computes_p50_p95_and_breach() -> None:
    """Rolling percentiles flag breach when p95 exceeds the SLA."""
    tracker = ModelLatencySlaTracker(window_size=20, sla_p95_ms=100.0)
    for latency in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 200.0):
        snap = tracker.record("gpt-5.5", latency)
    assert isinstance(snap, LatencySlaSnapshot)
    assert snap.samples == 10
    assert snap.p50_ms == pytest.approx(55.0)
    assert snap.p95_ms > 100.0
    assert snap.breached is True
    assert snap.sla_p95_ms == 100.0


def test_per_model_sla_override() -> None:
    """Per-model SLA overrides the default threshold."""
    tracker = ModelLatencySlaTracker(
        window_size=10,
        sla_p95_ms=50.0,
        per_model_sla_p95_ms={"claude-sonnet-4-6": 500.0},
    )
    for latency in (100.0, 120.0, 140.0, 160.0):
        snap = tracker.record("claude-sonnet-4-6", latency)
    assert snap.breached is False
    assert tracker.sla_for("claude-sonnet-4-6") == 500.0
    assert tracker.sla_for("gpt-5.5") == 50.0


def test_window_size_evicts_oldest_samples() -> None:
    """Only the newest window_size samples influence percentiles."""
    tracker = ModelLatencySlaTracker(window_size=3, sla_p95_ms=1_000.0)
    tracker.record("gemini-3.5-flash", 5_000.0)
    tracker.record("gemini-3.5-flash", 5_000.0)
    tracker.record("gemini-3.5-flash", 5_000.0)
    assert tracker.snapshot("gemini-3.5-flash") is not None
    assert tracker.snapshot("gemini-3.5-flash").breached is True  # type: ignore[union-attr]
    tracker.record("gemini-3.5-flash", 10.0)
    tracker.record("gemini-3.5-flash", 12.0)
    tracker.record("gemini-3.5-flash", 14.0)
    snap = tracker.snapshot("gemini-3.5-flash")
    assert snap is not None
    assert snap.samples == 3
    assert snap.breached is False
    assert snap.p95_ms < 20.0


def test_breaches_lists_only_violating_models() -> None:
    """breaches() returns sorted snapshots for models over their p95 SLA."""
    tracker = ModelLatencySlaTracker(window_size=10, sla_p95_ms=100.0)
    tracker.record("kimi-k2", 20.0)
    tracker.record("kimi-k2", 25.0)
    for latency in (150.0, 160.0, 170.0, 180.0):
        tracker.record("gpt-5.5", latency)
    breached = tracker.breaches()
    assert [item.model for item in breached] == ["gpt-5.5"]
    assert breached[0].breached is True


def test_models_clear_and_unknown_snapshot() -> None:
    """models()/clear()/snapshot(None-path) behave as an in-memory store."""
    clock = _Clock(10.0)
    tracker = ModelLatencySlaTracker(window_size=5, sla_p95_ms=200.0, clock=clock)
    assert tracker.snapshot("missing") is None
    tracker.record("gpt-5.5", 40.0)
    tracker.record("kimi-k2", 50.0)
    assert tracker.models() == ["gpt-5.5", "kimi-k2"]
    tracker.clear("gpt-5.5")
    assert tracker.models() == ["kimi-k2"]
    tracker.clear()
    assert tracker.models() == []


def test_rejects_invalid_inputs() -> None:
    """Construction and record validate window, SLA, model, and latency."""
    with pytest.raises(ValueError, match="window_size"):
        ModelLatencySlaTracker(window_size=0)
    with pytest.raises(ValueError, match="sla_p95_ms"):
        ModelLatencySlaTracker(sla_p95_ms=0)
    with pytest.raises(ValueError, match="per-model"):
        ModelLatencySlaTracker(per_model_sla_p95_ms={"": 10.0})
    with pytest.raises(ValueError, match="per-model"):
        ModelLatencySlaTracker(per_model_sla_p95_ms={"gpt-5.5": -1.0})
    tracker = ModelLatencySlaTracker()
    with pytest.raises(ValueError, match="model"):
        tracker.record("", 10.0)
    with pytest.raises(ValueError, match="latency_ms"):
        tracker.record("gpt-5.5", -1.0)
    with pytest.raises(ValueError, match="model"):
        tracker.sla_for("")
