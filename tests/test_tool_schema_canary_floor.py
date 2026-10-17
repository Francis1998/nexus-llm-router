"""Tests for ToolSchemaCanaryFloorAdvisor."""

from __future__ import annotations

import pytest

from safety.tool_schema_canary_floor import ToolSchemaCanaryFloorAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = ToolSchemaCanaryFloorAdvisor().advise(request_id="r1", canary_miss_ratio=0.01)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = ToolSchemaCanaryFloorAdvisor().advise(
        request_id="r1",
        canary_miss_ratio=0.15,
        soft_limit=0.15,
        hard_limit=0.4,
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = ToolSchemaCanaryFloorAdvisor().advise(
        request_id="r1",
        canary_miss_ratio=0.4,
        soft_limit=0.15,
        hard_limit=0.4,
    )
    assert advice.band == "breach"


def test_invalid() -> None:
    """Invalid inputs raise."""

    with pytest.raises(ValueError, match="request_id"):
        ToolSchemaCanaryFloorAdvisor().advise(request_id=" ", canary_miss_ratio=0.1)
