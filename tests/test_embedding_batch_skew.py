"""Unit tests for EmbeddingBatchSkewAdvisor."""

from __future__ import annotations

import pytest

from safety.embedding_batch_skew import EmbeddingBatchSkewAdvisor


def test_within_band() -> None:
    """Low skew is within."""

    advice = EmbeddingBatchSkewAdvisor().advise(request_id="r1", batch_size=8, target_size=8)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Mid skew is soft."""

    advice = EmbeddingBatchSkewAdvisor().advise(request_id="r1", batch_size=16, target_size=8)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """High skew is breach."""

    advice = EmbeddingBatchSkewAdvisor().advise(request_id="r1", batch_size=40, target_size=8)
    assert advice.band == "breach"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        EmbeddingBatchSkewAdvisor().advise(request_id=" ", batch_size=1, target_size=1)
