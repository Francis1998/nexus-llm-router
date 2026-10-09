"""Unit tests for EmbeddingCacheThrashAdvisor."""

from __future__ import annotations

import pytest

from safety.embedding_cache_thrash import EmbeddingCacheThrashAdvisor


def test_within() -> None:
    """Band within."""

    advice = EmbeddingCacheThrashAdvisor().advise(request_id="r1", thrash_ratio=0.15)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = EmbeddingCacheThrashAdvisor().advise(request_id="r1", thrash_ratio=0.5)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = EmbeddingCacheThrashAdvisor().advise(request_id="r1", thrash_ratio=1.7)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="thrash_ratio"):
        EmbeddingCacheThrashAdvisor().advise(request_id="r1", thrash_ratio=-0.1)
