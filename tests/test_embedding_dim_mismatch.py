"""Unit tests for EmbeddingDimMismatchAdvisor."""

from __future__ import annotations

import pytest

from safety.embedding_dim_mismatch import EmbeddingDimMismatchAdvisor


def test_within() -> None:
    """Low metric is within."""

    advice = EmbeddingDimMismatchAdvisor().advise(request_id="r1", dim_delta=0.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = EmbeddingDimMismatchAdvisor().advise(request_id="r1", dim_delta=2.0)
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = EmbeddingDimMismatchAdvisor().advise(request_id="r1", dim_delta=16.0)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="dim_delta"):
        EmbeddingDimMismatchAdvisor().advise(request_id="r1", dim_delta=-0.1)
