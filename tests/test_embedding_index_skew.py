"""Tests for EmbeddingIndexSkewAdvisor."""

from __future__ import annotations

import pytest

from safety.embedding_index_skew import EmbeddingIndexSkewAdvisor


def test_within() -> None:
    """Band within."""

    advice = EmbeddingIndexSkewAdvisor().advise(request_id="r1", index_skew_ratio=0.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.2 + 0.5) / 2
    advice = EmbeddingIndexSkewAdvisor().advise(request_id="r1", index_skew_ratio=mid)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = EmbeddingIndexSkewAdvisor().advise(request_id="r1", index_skew_ratio=1.0000)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError, match="request_id"):
        EmbeddingIndexSkewAdvisor().advise(request_id=" ", index_skew_ratio=0.0)
