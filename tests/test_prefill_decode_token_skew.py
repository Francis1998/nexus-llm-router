"""Unit tests for PrefillDecodeTokenSkewAdvisor."""

from __future__ import annotations

import pytest

from safety.prefill_decode_token_skew import PrefillDecodeTokenSkewAdvisor


def test_within() -> None:
    """Low metric is within."""

    advice = PrefillDecodeTokenSkewAdvisor().advise(request_id="r1", skew_ratio=0.8)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = PrefillDecodeTokenSkewAdvisor().advise(request_id="r1", skew_ratio=3.5)
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = PrefillDecodeTokenSkewAdvisor().advise(request_id="r1", skew_ratio=6.0)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="skew_ratio"):
        PrefillDecodeTokenSkewAdvisor().advise(request_id="r1", skew_ratio=-0.1)
