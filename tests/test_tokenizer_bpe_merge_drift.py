"""Tests for TokenizerBpeMergeDriftAdvisor."""

from __future__ import annotations

import pytest

from safety.tokenizer_bpe_merge_drift import TokenizerBpeMergeDriftAdvisor


def test_within() -> None:
    """Band within."""

    advice = TokenizerBpeMergeDriftAdvisor().advise(request_id="r1", bpe_merge_drift_score=0.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.2 + 0.5) / 2
    advice = TokenizerBpeMergeDriftAdvisor().advise(request_id="r1", bpe_merge_drift_score=mid)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = TokenizerBpeMergeDriftAdvisor().advise(request_id="r1", bpe_merge_drift_score=1.0000)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError, match="request_id"):
        TokenizerBpeMergeDriftAdvisor().advise(request_id=" ", bpe_merge_drift_score=0.0)
