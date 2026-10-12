"""Unit tests for ToolSchemaVersionSkewAdvisor."""

from __future__ import annotations

import pytest

from safety.tool_schema_version_skew import ToolSchemaVersionSkewAdvisor


def test_within() -> None:
    """Band within."""

    advice = ToolSchemaVersionSkewAdvisor().advise(request_id="r1", skew_score=0.15)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = ToolSchemaVersionSkewAdvisor().advise(request_id="r1", skew_score=0.5)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = ToolSchemaVersionSkewAdvisor().advise(request_id="r1", skew_score=1.7)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="skew_score"):
        ToolSchemaVersionSkewAdvisor().advise(request_id="r1", skew_score=-0.1)
