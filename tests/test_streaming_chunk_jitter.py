"""Unit tests for StreamingChunkJitterSloAdvisor."""

from __future__ import annotations

import pytest

from safety.streaming_chunk_jitter import StreamingChunkJitterSloAdvisor


def test_within_band() -> None:
    """At/under SLO is within."""

    advice = StreamingChunkJitterSloAdvisor().advise(request_id="r1", jitter_ms=40.0, slo_ms=50.0)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Up to 1.5x SLO is soft."""

    advice = StreamingChunkJitterSloAdvisor().advise(request_id="r1", jitter_ms=60.0, slo_ms=50.0)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """Over 1.5x SLO is breach."""

    advice = StreamingChunkJitterSloAdvisor().advise(request_id="r1", jitter_ms=100.0, slo_ms=50.0)
    assert advice.band == "breach"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        StreamingChunkJitterSloAdvisor().advise(request_id=" ", jitter_ms=10.0, slo_ms=50.0)
