"""Unit tests for JsonSchemaStrictnessBandAdvisor."""

from __future__ import annotations

import pytest

from safety.json_schema_strictness_band import JsonSchemaStrictnessBandAdvisor


def test_within() -> None:
    """Band within."""

    advice = JsonSchemaStrictnessBandAdvisor().advise(request_id="r1", strictness_score=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = JsonSchemaStrictnessBandAdvisor().advise(request_id="r1", strictness_score=0.45)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = JsonSchemaStrictnessBandAdvisor().advise(request_id="r1", strictness_score=0.9)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="strictness_score"):
        JsonSchemaStrictnessBandAdvisor().advise(request_id="r1", strictness_score=-0.1)
