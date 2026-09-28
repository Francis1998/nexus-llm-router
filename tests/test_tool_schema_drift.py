"""Unit tests for ToolSchemaDriftGate."""

from __future__ import annotations

import pytest

from safety.tool_schema_drift import ToolSchemaDriftGate


def test_stable_band() -> None:
    """Identical hashes are stable."""

    advice = ToolSchemaDriftGate().advise(
        tool_name="search", declared_hash="abc123", observed_hash="abc123"
    )
    assert advice.band == "stable"


def test_soft_drift_band() -> None:
    """Shared 8-char prefix is soft_drift."""

    advice = ToolSchemaDriftGate().advise(
        tool_name="search",
        declared_hash="abc12345xxxx",
        observed_hash="abc12345yyyy",
    )
    assert advice.band == "soft_drift"


def test_hard_drift_band() -> None:
    """Unrelated hashes are hard_drift."""

    advice = ToolSchemaDriftGate().advise(
        tool_name="search", declared_hash="aaaaaaaa", observed_hash="bbbbbbbb"
    )
    assert advice.band == "hard_drift"


def test_empty_tool_raises() -> None:
    """Empty tool name raises ValueError."""

    with pytest.raises(ValueError, match="tool_name"):
        ToolSchemaDriftGate().advise(tool_name=" ", declared_hash="a", observed_hash="a")
