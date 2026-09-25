"""Unit tests for ParallelToolCallArityAdvisor."""

from __future__ import annotations

import pytest

from safety.parallel_tool_call_arity import ParallelToolCallArityAdvisor


def test_ok_band() -> None:
    """Below soft limit is ok."""

    advice = ParallelToolCallArityAdvisor().advise(tool_call_count=2)
    assert advice.band == "ok"


def test_busy_band() -> None:
    """At/above soft and below hard is busy."""

    advice = ParallelToolCallArityAdvisor().advise(tool_call_count=4)
    assert advice.band == "busy"


def test_fanout_band() -> None:
    """At/above hard limit is fanout."""

    advice = ParallelToolCallArityAdvisor().advise(tool_call_count=12)
    assert advice.band == "fanout"


def test_invalid_limits_raise() -> None:
    """hard_limit < soft_limit raises ValueError."""

    with pytest.raises(ValueError, match="hard_limit"):
        ParallelToolCallArityAdvisor().advise(
            tool_call_count=1,
            soft_limit=8,
            hard_limit=4,
        )
