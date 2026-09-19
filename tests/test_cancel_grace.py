"""Tests for StreamingCancelGraceGuard open/grace/closed bands."""

from __future__ import annotations

import pytest

from safety.cancel_grace import StreamingCancelGraceGuard


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_open_then_grace() -> None:
    clock = _Clock()
    guard = StreamingCancelGraceGuard(grace_ms=100.0, clock=clock)
    snap = guard.cancel("req-1")
    assert snap.band == "open"
    assert snap.allow_final_chunk is True
    clock.t = 0.05
    mid = guard.snapshot("req-1")
    assert mid is not None
    assert mid.band == "grace"
    assert mid.allow_final_chunk is True


def test_closed_after_grace() -> None:
    clock = _Clock()
    guard = StreamingCancelGraceGuard(grace_ms=100.0, clock=clock)
    guard.cancel("req-2")
    clock.t = 0.2
    snap = guard.snapshot("req-2")
    assert snap is not None
    assert snap.band == "closed"
    assert snap.allow_final_chunk is False
    assert guard.allow_final_chunk("req-2") is False


def test_unknown_allows_final() -> None:
    guard = StreamingCancelGraceGuard(grace_ms=50.0)
    assert guard.allow_final_chunk("never-cancelled") is True


def test_clear() -> None:
    clock = _Clock()
    guard = StreamingCancelGraceGuard(grace_ms=50.0, clock=clock)
    guard.cancel("req-3")
    guard.clear("req-3")
    assert guard.snapshot("req-3") is None


def test_invalid_grace_raises() -> None:
    with pytest.raises(ValueError, match="grace_ms"):
        StreamingCancelGraceGuard(grace_ms=0)
    with pytest.raises(ValueError, match="request_id"):
        StreamingCancelGraceGuard().cancel("")
