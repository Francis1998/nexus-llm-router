"""Tests for StreamingBackpressureAdvisor inter-chunk gap bands."""

from __future__ import annotations

import pytest

from safety.stream_backpressure import StreamBackpressureAdvice, StreamingBackpressureAdvisor


def test_advise_ok_stall_and_backpressure_bands() -> None:
    """Inter-chunk gaps map to ok / stall / backpressure relative to thresholds."""
    advisor = StreamingBackpressureAdvisor(
        stall_threshold_ms=200.0,
        backpressure_threshold_ms=800.0,
    )
    ok = advisor.advise(stream_id="gpt-5.5-s1", gap_ms=50.0)
    assert isinstance(ok, StreamBackpressureAdvice)
    assert ok.band == "ok"
    stall = advisor.advise(stream_id="claude-sonnet-4-6-s1", gap_ms=250.0)
    assert stall.band == "stall"
    pressure = advisor.advise(stream_id="gemini-3-s1", gap_ms=900.0)
    assert pressure.band == "backpressure"
    assert "backpressure" in pressure.advisory


def test_record_tracks_rolling_gaps_and_backpressured() -> None:
    """record() keeps a rolling window; backpressured() lists stalled streams."""
    advisor = StreamingBackpressureAdvisor(
        stall_threshold_ms=100.0,
        backpressure_threshold_ms=300.0,
        window_size=5,
    )
    advisor.record("gpt-5.5-stream", 40.0)
    for gap in (350.0, 400.0, 450.0):
        advisor.record("kimi-k2-stream", gap)
    listed = advisor.backpressured()
    assert [item.stream_id for item in listed] == ["kimi-k2-stream"]
    assert listed[0].band == "backpressure"
    assert "gpt-5.5-stream" in advisor.streams()


def test_window_evicts_oldest_gap_samples() -> None:
    """Only the newest window_size gap samples drive the rolling snapshot."""
    advisor = StreamingBackpressureAdvisor(
        stall_threshold_ms=100.0,
        backpressure_threshold_ms=200.0,
        window_size=3,
    )
    for _ in range(3):
        advisor.record("gemini-3.5-flash", 5_000.0)
    snap = advisor.snapshot("gemini-3.5-flash")
    assert snap is not None
    assert snap.band == "backpressure"
    for gap in (10.0, 12.0, 14.0):
        advisor.record("gemini-3.5-flash", gap)
    snap = advisor.snapshot("gemini-3.5-flash")
    assert snap is not None
    assert snap.samples == 3
    assert snap.band == "ok"
    assert snap.p50_gap_ms == pytest.approx(12.0)


def test_frontier_stream_ids_gpt_claude_gemini_kimi() -> None:
    """Frontier model stream ids for GPT-5.5 / Sonnet 4.6 / Gemini 3.x / Kimi K2."""
    advisor = StreamingBackpressureAdvisor()
    advisor.record("gpt-5.5", 20.0)
    advisor.record("claude-sonnet-4-6", 30.0)
    advisor.record("gemini-3.x", 40.0)
    advisor.record("kimi-k2", 50.0)
    assert advisor.streams() == [
        "gpt-5.5",
        "claude-sonnet-4-6",
        "gemini-3.x",
        "kimi-k2",
    ]


def test_rejects_invalid_inputs() -> None:
    """Construction and advise/record validate thresholds and inputs."""
    with pytest.raises(ValueError, match="stall_threshold_ms"):
        StreamingBackpressureAdvisor(stall_threshold_ms=0)
    with pytest.raises(ValueError, match="backpressure_threshold_ms"):
        StreamingBackpressureAdvisor(
            stall_threshold_ms=100.0,
            backpressure_threshold_ms=100.0,
        )
    with pytest.raises(ValueError, match="window_size"):
        StreamingBackpressureAdvisor(window_size=0)
    advisor = StreamingBackpressureAdvisor()
    with pytest.raises(ValueError, match="stream_id"):
        advisor.advise(stream_id="", gap_ms=10.0)
    with pytest.raises(ValueError, match="gap_ms"):
        advisor.advise(stream_id="s1", gap_ms=-1.0)
    assert advisor.snapshot("missing") is None
    advisor.clear()
    assert advisor.streams() == []
