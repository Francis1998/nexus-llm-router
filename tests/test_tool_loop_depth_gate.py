"""Unit tests for ToolLoopDepthGateAdvisor."""

from __future__ import annotations

import pytest

from safety.tool_loop_depth_gate import ToolLoopDepthGateAdvisor


def test_within() -> None:
    """Band within."""

    advice = ToolLoopDepthGateAdvisor().advise(request_id="r1", loop_depth=1.5)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = ToolLoopDepthGateAdvisor().advise(request_id="r1", loop_depth=5.5)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = ToolLoopDepthGateAdvisor().advise(request_id="r1", loop_depth=9.0)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="loop_depth"):
        ToolLoopDepthGateAdvisor().advise(request_id="r1", loop_depth=-0.1)
