"""Tests for OutputJsonSchemaGuard required-keys JSON validation."""

from __future__ import annotations

import pytest

from safety.output_schema_guard import OutputJsonSchemaGuard, OutputSchemaVerdict


def test_ok_when_all_required_keys_present() -> None:
    """Valid JSON objects with every required key yield verdict ok."""
    guard = OutputJsonSchemaGuard(required_keys=["model", "answer", "confidence"])
    verdict = guard.check('{"model":"gpt-5.5","answer":"yes","confidence":0.9}')
    assert isinstance(verdict, OutputSchemaVerdict)
    assert verdict.verdict == "ok"
    assert verdict.missing_keys == []
    assert verdict.parsed is not None
    assert verdict.parsed["model"] == "gpt-5.5"
    assert verdict.reason is None


def test_missing_reports_absent_keys() -> None:
    """JSON objects missing required keys yield verdict missing."""
    guard = OutputJsonSchemaGuard(required_keys=["answer", "citations", "model"])
    verdict = guard.check('{"answer":"ok","model":"claude-sonnet-4.6"}')
    assert verdict.verdict == "missing"
    assert verdict.missing_keys == ["citations"]
    assert verdict.parsed is not None
    assert "citations" in (verdict.reason or "")


def test_invalid_json_when_payload_not_parseable() -> None:
    """Non-JSON / non-object payloads yield verdict invalid_json."""
    guard = OutputJsonSchemaGuard(required_keys=["answer"])
    bad = guard.check("not-json-at-all")
    assert bad.verdict == "invalid_json"
    assert bad.parsed is None
    assert bad.missing_keys == []

    array = guard.check("[1,2,3]")
    assert array.verdict == "invalid_json"
    assert array.parsed is None


def test_frontier_model_fields_validated_consistently() -> None:
    """Same schema applies for GPT-5.5 / Sonnet / Gemini / Kimi shaped outputs."""
    guard = OutputJsonSchemaGuard(required_keys=["provider", "model_id"])
    payloads = (
        '{"provider":"openai","model_id":"gpt-5.5"}',
        '{"provider":"anthropic","model_id":"claude-sonnet-4.6"}',
        '{"provider":"google","model_id":"gemini-3.1-pro-preview"}',
        '{"provider":"moonshot","model_id":"kimi-k2"}',
    )
    for text in payloads:
        verdict = guard.check(text)
        assert verdict.verdict == "ok"
        assert verdict.parsed is not None


def test_rejects_empty_required_keys_and_duplicate_config() -> None:
    """Constructor requires a non-empty unique required_keys list."""
    with pytest.raises(ValueError, match="required_keys"):
        OutputJsonSchemaGuard(required_keys=[])
    with pytest.raises(ValueError, match="required_keys"):
        OutputJsonSchemaGuard(required_keys=["a", ""])
    with pytest.raises(ValueError, match="duplicate"):
        OutputJsonSchemaGuard(required_keys=["a", "a"])
