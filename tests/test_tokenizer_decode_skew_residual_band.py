"""Tests for TokenizerDecodeSkewResidualBandAdvisor."""

from __future__ import annotations

import pytest

from safety.tokenizer_decode_skew_residual_band import TokenizerDecodeSkewResidualBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = TokenizerDecodeSkewResidualBandAdvisor().advise(request_id="r1", decode_skew_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = TokenizerDecodeSkewResidualBandAdvisor().advise(request_id="r1", decode_skew_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = TokenizerDecodeSkewResidualBandAdvisor().advise(request_id="r1", decode_skew_ratio=0.9)
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        TokenizerDecodeSkewResidualBandAdvisor().advise(request_id=" ", decode_skew_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="decode_skew_ratio"):
        TokenizerDecodeSkewResidualBandAdvisor().advise(request_id="r1", decode_skew_ratio=-0.1)
