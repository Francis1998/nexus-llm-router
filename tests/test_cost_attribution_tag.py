"""Tests for CostAttributionTagLedger."""

from __future__ import annotations

import pytest

from safety.cost_attribution_tag import CostAttributionTagLedger


def test_record_and_rollup() -> None:
    ledger = CostAttributionTagLedger()
    ledger.record("r1", tags=("team:a", "env:prod"), cost_usd=0.02)
    ledger.record("r2", tags=("team:a",), cost_usd=0.03)
    rows = ledger.rollup()
    by_tag = {r.tag: r for r in rows}
    assert by_tag["team:a"].total_cost_usd == 0.05
    assert by_tag["team:a"].request_count == 2
    assert by_tag["env:prod"].request_count == 1


def test_dedupes_duplicate_tags() -> None:
    entry = CostAttributionTagLedger().record("r1", tags=("team:a", "team:a"), cost_usd=0.01)
    assert entry.tags == ("team:a",)


def test_invalid() -> None:
    with pytest.raises(ValueError, match="request_id"):
        CostAttributionTagLedger().record("", tags=("a",), cost_usd=0.1)
    with pytest.raises(ValueError, match="tags"):
        CostAttributionTagLedger().record("r", tags=("  ",), cost_usd=0.1)
    with pytest.raises(ValueError, match="cost_usd"):
        CostAttributionTagLedger().record("r", tags=("a",), cost_usd=-1)
