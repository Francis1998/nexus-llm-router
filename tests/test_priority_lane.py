"""Tests for RequestPriorityLane weighted fair dequeue."""

from __future__ import annotations

import pytest

from safety.priority_lane import (
    LaneDepthSnapshot,
    PriorityClass,
    QueuedRequest,
    RequestPriorityLane,
)


def test_enqueue_and_depth_by_lane() -> None:
    """Requests land in the correct lane and depths aggregate."""
    lane = RequestPriorityLane()
    lane.enqueue("h1", PriorityClass.HIGH)
    lane.enqueue("n1", "normal")
    lane.enqueue("b1", "bulk")
    lane.enqueue("n2", PriorityClass.NORMAL)
    assert lane.depth(PriorityClass.HIGH) == 1
    assert lane.depth("normal") == 2
    assert lane.depth("bulk") == 1
    assert lane.depth() == 4
    snap = lane.snapshot()
    assert isinstance(snap, LaneDepthSnapshot)
    assert snap == LaneDepthSnapshot(high=1, normal=2, bulk=1, total=4)


def test_fair_dequeue_prefers_high_without_starving_bulk() -> None:
    """Default 4:2:1 weights serve high more often but still drain bulk."""
    lane = RequestPriorityLane()
    for index in range(4):
        lane.enqueue(f"h{index}", "high")
    for index in range(2):
        lane.enqueue(f"n{index}", "normal")
    lane.enqueue("b0", "bulk")

    served = [lane.dequeue() for _ in range(7)]
    assert all(isinstance(item, QueuedRequest) for item in served)
    priorities = [item.priority for item in served if item is not None]
    assert priorities.count(PriorityClass.HIGH) == 4
    assert priorities.count(PriorityClass.NORMAL) == 2
    assert priorities.count(PriorityClass.BULK) == 1
    # First slot in the default cycle is high.
    assert served[0] is not None and served[0].priority == PriorityClass.HIGH
    assert lane.dequeue() is None
    assert lane.depth() == 0


def test_empty_lanes_are_skipped_in_cycle() -> None:
    """Quiet high lane does not block normal/bulk service."""
    lane = RequestPriorityLane()
    lane.enqueue("n1", PriorityClass.NORMAL)
    lane.enqueue("b1", PriorityClass.BULK)
    first = lane.dequeue()
    second = lane.dequeue()
    assert first is not None and first.request_id == "n1"
    assert second is not None and second.request_id == "b1"


def test_custom_weights_and_pending_order() -> None:
    """Custom weights reshape the cycle; pending lists high then normal then bulk."""
    lane = RequestPriorityLane(weights={"high": 1, "normal": 1, "bulk": 1})
    assert lane.weights[PriorityClass.HIGH] == 1
    lane.enqueue("b1", "bulk")
    lane.enqueue("h1", "high")
    lane.enqueue("n1", "normal")
    pending = lane.pending()
    assert [item.request_id for item in pending] == ["h1", "n1", "b1"]
    order = [lane.dequeue() for _ in range(3)]
    assert [item.request_id for item in order if item is not None] == ["h1", "n1", "b1"]


def test_clear_resets_queues() -> None:
    """clear() drops pending work and resets the cycle cursor."""
    lane = RequestPriorityLane()
    lane.enqueue("h1", "high")
    lane.enqueue("b1", "bulk")
    lane.clear()
    assert lane.snapshot().total == 0
    assert lane.dequeue() is None


def test_rejects_invalid_inputs() -> None:
    """Empty request ids, unknown priorities, and non-positive weights fail."""
    with pytest.raises(ValueError, match="weight"):
        RequestPriorityLane(weights={"high": 0})
    lane = RequestPriorityLane()
    with pytest.raises(ValueError, match="request_id"):
        lane.enqueue("", "high")
    with pytest.raises(ValueError, match="priority"):
        lane.enqueue("r1", "urgent")


def test_frontier_model_request_ids_round_trip() -> None:
    """Lane scheduling is model-agnostic for GPT-5.5 / Sonnet 4.6 / Gemini / Kimi."""
    lane = RequestPriorityLane()
    lane.enqueue("gpt-5.5-interactive", "high")
    lane.enqueue("claude-sonnet-4-6-chat", "normal")
    lane.enqueue("gemini-3-batch", "bulk")
    lane.enqueue("kimi-k2-bulk", "bulk")
    served_ids = []
    while True:
        item = lane.dequeue()
        if item is None:
            break
        served_ids.append(item.request_id)
    assert "gpt-5.5-interactive" in served_ids
    assert served_ids[0] == "gpt-5.5-interactive"
    assert set(served_ids) == {
        "gpt-5.5-interactive",
        "claude-sonnet-4-6-chat",
        "gemini-3-batch",
        "kimi-k2-bulk",
    }
