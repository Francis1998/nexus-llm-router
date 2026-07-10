"""Tests for provider HTTP-response parsing helpers."""

from __future__ import annotations

from adapters.http_utils import nested_int


def test_nested_int_reads_plain_integer() -> None:
    """A plain integer field is returned unchanged."""
    payload = {"usage": {"prompt_tokens": 12}}

    assert nested_int(payload, ["usage", "prompt_tokens"]) == 12


def test_nested_int_coerces_integral_float() -> None:
    """An integral float usage count must be coerced, not dropped to the default.

    Some OpenAI-compatible gateways serialize usage token counts as JSON numbers
    with a decimal point (for example ``12.0``), which ``json`` decodes to a
    Python ``float``. The previous ``isinstance(value, int)`` check treated that
    as missing and returned ``0``, silently zeroing token accounting, the cost
    estimate, and the audit record. Integral floats must be coerced to ``int``.
    """
    payload = {"usage": {"prompt_tokens": 12.0}}

    assert nested_int(payload, ["usage", "prompt_tokens"]) == 12


def test_nested_int_ignores_non_integral_float() -> None:
    """A non-integral float is not a valid token count and yields the default."""
    payload = {"usage": {"prompt_tokens": 12.5}}

    assert nested_int(payload, ["usage", "prompt_tokens"]) == 0


def test_nested_int_returns_default_for_missing_path() -> None:
    """A missing nested path returns the supplied default."""
    assert nested_int({}, ["usage", "prompt_tokens"], default=-1) == -1


def test_nested_int_coerces_integer_string() -> None:
    """An integer-valued string usage count must be coerced, not dropped.

    Some OpenAI-compatible gateways serialize usage token counts as JSON
    strings (for example ``"12"``). The previous checks treated a string as
    missing and returned the default, silently zeroing token accounting, the
    cost estimate, and the audit record. A whitespace-trimmed non-negative
    integer string must be coerced to ``int``.
    """
    payload = {"usage": {"prompt_tokens": "  12  "}}

    assert nested_int(payload, ["usage", "prompt_tokens"]) == 12


def test_nested_int_ignores_non_numeric_string() -> None:
    """A non-numeric string is not a valid token count and yields the default."""
    payload = {"usage": {"prompt_tokens": "not-a-number"}}

    assert nested_int(payload, ["usage", "prompt_tokens"]) == 0
