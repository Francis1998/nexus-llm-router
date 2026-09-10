"""Tests for StreamingTokenBudgetGate hard mid-stream cut-off."""

from __future__ import annotations

import pytest

from safety.stream_budget import (
    StreamBudgetSnapshot,
    StreamingTokenBudgetExceededError,
    StreamingTokenBudgetGate,
)


def test_token_budget_exhausts_mid_stream() -> None:
    """Chunks accumulate until the hard token ceiling trips."""
    gate = StreamingTokenBudgetGate(max_tokens=10)
    first = gate.observe_chunk(tokens=4)
    assert first.exhausted is False
    assert first.tokens_used == 4
    assert gate.remaining_tokens() == 6
    assert gate.allow_continue() is True

    second = gate.observe_chunk(tokens=6)
    assert second.exhausted is True
    assert second.tokens_used == 10
    assert second.reason is not None and "token budget exhausted" in second.reason
    assert gate.allow_continue() is False
    assert gate.remaining_tokens() == 0


def test_cost_budget_hard_stop_raises() -> None:
    """hard_stop=True raises when cumulative cost crosses the ceiling."""
    gate = StreamingTokenBudgetGate(max_cost_usd=0.01)
    gate.observe_chunk(tokens=5, cost_usd=0.004)
    with pytest.raises(StreamingTokenBudgetExceededError) as exc_info:
        gate.observe_chunk(tokens=5, cost_usd=0.007, hard_stop=True)
    snapshot = exc_info.value.snapshot
    assert isinstance(snapshot, StreamBudgetSnapshot)
    assert snapshot.exhausted is True
    assert snapshot.cost_usd == pytest.approx(0.011)
    assert snapshot.reason is not None and "cost budget exhausted" in snapshot.reason


def test_assert_within_budget_after_exhaustion() -> None:
    """assert_within_budget raises once a prior chunk exhausted the gate."""
    gate = StreamingTokenBudgetGate(max_tokens=3, max_cost_usd=1.0)
    gate.observe_chunk(tokens=3)
    with pytest.raises(StreamingTokenBudgetExceededError):
        gate.assert_within_budget()


def test_further_chunks_do_not_increase_usage_after_exhaustion() -> None:
    """Once exhausted, additional observe_chunk calls are no-ops for counters."""
    gate = StreamingTokenBudgetGate(max_tokens=5)
    gate.observe_chunk(tokens=5)
    after = gate.observe_chunk(tokens=20, cost_usd=9.0)
    assert after.tokens_used == 5
    assert after.cost_usd == 0.0
    assert after.exhausted is True


def test_reset_clears_usage_for_reuse() -> None:
    """reset() clears counters so the gate can guard another stream."""
    gate = StreamingTokenBudgetGate(max_tokens=4, max_cost_usd=0.5)
    gate.observe_chunk(tokens=4, cost_usd=0.1)
    assert gate.allow_continue() is False
    gate.reset()
    snap = gate.snapshot()
    assert snap.tokens_used == 0
    assert snap.cost_usd == 0.0
    assert snap.exhausted is False
    assert snap.reason is None
    assert gate.allow_continue() is True


def test_rejects_invalid_construction_and_chunk_inputs() -> None:
    """Construction and observe_chunk validate budgets and deltas."""
    with pytest.raises(ValueError, match="at least one"):
        StreamingTokenBudgetGate()
    with pytest.raises(ValueError, match="max_tokens"):
        StreamingTokenBudgetGate(max_tokens=0)
    with pytest.raises(ValueError, match="max_cost_usd"):
        StreamingTokenBudgetGate(max_cost_usd=-0.1)
    gate = StreamingTokenBudgetGate(max_tokens=10)
    with pytest.raises(ValueError, match="tokens"):
        gate.observe_chunk(tokens=-1)
    with pytest.raises(ValueError, match="cost_usd"):
        gate.observe_chunk(cost_usd=-0.01)


def test_frontier_model_labels_covered_in_snapshot_fields() -> None:
    """Gate budgets apply equally to GPT-5.5 / Sonnet 4.6 / Gemini 3.x / Kimi K2."""
    gate = StreamingTokenBudgetGate(max_tokens=100, max_cost_usd=0.25)
    for model_tokens, cost in (
        (12, 0.02),  # gpt-5.5
        (18, 0.03),  # claude-sonnet-4-6
        (9, 0.01),  # gemini-3.x
        (7, 0.005),  # kimi-k2
    ):
        snap = gate.observe_chunk(tokens=model_tokens, cost_usd=cost)
        assert snap.exhausted is False
    assert gate.snapshot().tokens_used == 46
    assert gate.remaining_cost_usd() == pytest.approx(0.185)
