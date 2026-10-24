"""Unit tests for TokenizerVocabSkewBandAdvisor."""

from __future__ import annotations

import pytest

from safety.tokenizer_vocab_skew_band import (
    TokenizerVocabSkewBandAdvisor,
)


def test_within() -> None:
    """Below soft -> within."""

    advice = TokenizerVocabSkewBandAdvisor().advise(request_id="r1", vocab_skew_ratio=0.05)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard -> soft."""

    advice = TokenizerVocabSkewBandAdvisor().advise(request_id="r1", vocab_skew_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard -> breach."""

    advice = TokenizerVocabSkewBandAdvisor().advise(request_id="r1", vocab_skew_ratio=0.8)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        TokenizerVocabSkewBandAdvisor().advise(request_id=" ", vocab_skew_ratio=0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        TokenizerVocabSkewBandAdvisor().advise(
            request_id="r1",
            vocab_skew_ratio=0.1,
            soft_limit=0.5,
            hard_limit=0.2,
        )
