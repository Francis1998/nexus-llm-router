"""Tests for StructuredOutputRepairAdvisor."""

from __future__ import annotations

import pytest

from safety.structured_output_repair import StructuredOutputRepairAdvisor


def test_accept_complete_object() -> None:
    advice = StructuredOutputRepairAdvisor().advise(
        "r1",
        payload='{"a": 1, "b": 2}',
        required_keys=("a", "b"),
    )
    assert advice.band == "accept"
    assert advice.missing_keys == ()


def test_repair_partial_object() -> None:
    advice = StructuredOutputRepairAdvisor().advise(
        "r2",
        payload='{"a": 1}',
        required_keys=("a", "b"),
    )
    assert advice.band == "repair"
    assert advice.missing_keys == ("b",)


def test_fail_invalid_json() -> None:
    advice = StructuredOutputRepairAdvisor().advise(
        "r3",
        payload="not-json",
        required_keys=("a",),
    )
    assert advice.band == "fail"
    assert advice.parse_ok is False


def test_fail_too_many_missing() -> None:
    advice = StructuredOutputRepairAdvisor().advise(
        "r4",
        payload='{"a": 1}',
        required_keys=("a", "b", "c", "d"),
    )
    assert advice.band == "fail"


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="request_id"):
        StructuredOutputRepairAdvisor().advise("", payload="{}", required_keys=("a",))
    with pytest.raises(ValueError, match="required_keys"):
        StructuredOutputRepairAdvisor().advise("r", payload="{}", required_keys=())
