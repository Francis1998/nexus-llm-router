"""Unit tests for ToolArgSizeForesightAdvisor."""

from __future__ import annotations

import pytest

from safety.tool_arg_size_foresight import ToolArgSizeForesightAdvisor


def test_within_band() -> None:
    """Low utilization is within."""

    advice = ToolArgSizeForesightAdvisor().advise(request_id="r1", arg_bytes=100, budget_bytes=1000)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Mid utilization is soft."""

    advice = ToolArgSizeForesightAdvisor().advise(request_id="r1", arg_bytes=800, budget_bytes=1000)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """Over budget is breach."""

    advice = ToolArgSizeForesightAdvisor().advise(
        request_id="r1", arg_bytes=1200, budget_bytes=1000
    )
    assert advice.band == "breach"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        ToolArgSizeForesightAdvisor().advise(request_id=" ", arg_bytes=1, budget_bytes=10)
