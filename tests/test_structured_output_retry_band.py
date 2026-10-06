"""Unit tests for StructuredOutputRetryBandAdvisor."""

from __future__ import annotations

import pytest

from safety.structured_output_retry_band import StructuredOutputRetryBandAdvisor


def test_within() -> None:
    """Low metric is within."""

    advice = StructuredOutputRetryBandAdvisor().advise(request_id="r1", retry_count=1.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = StructuredOutputRetryBandAdvisor().advise(request_id="r1", retry_count=3.0)
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = StructuredOutputRetryBandAdvisor().advise(request_id="r1", retry_count=6.0)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="retry_count"):
        StructuredOutputRetryBandAdvisor().advise(request_id="r1", retry_count=-1.0)
