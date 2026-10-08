"""Unit tests for ReasoningTraceLeakAdvisor."""

from __future__ import annotations

import pytest

from safety.reasoning_trace_leak import ReasoningTraceLeakAdvisor


def test_within() -> None:
    """Band within."""

    advice = ReasoningTraceLeakAdvisor().advise(request_id="r1", leak_score=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = ReasoningTraceLeakAdvisor().advise(request_id="r1", leak_score=0.45)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = ReasoningTraceLeakAdvisor().advise(request_id="r1", leak_score=0.9)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="leak_score"):
        ReasoningTraceLeakAdvisor().advise(request_id="r1", leak_score=-0.1)
