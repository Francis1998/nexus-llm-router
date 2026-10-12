"""Unit tests for PrefillDecodeSkewAdvisor."""

from __future__ import annotations

import pytest

from safety.prefill_decode_skew import PrefillDecodeSkewAdvisor


def test_within() -> None:
    """Band within."""

    advice = PrefillDecodeSkewAdvisor().advise(request_id="r1", skew_ratio=0.15)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = PrefillDecodeSkewAdvisor().advise(request_id="r1", skew_ratio=0.5)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = PrefillDecodeSkewAdvisor().advise(request_id="r1", skew_ratio=1.7)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="skew_ratio"):
        PrefillDecodeSkewAdvisor().advise(request_id="r1", skew_ratio=-0.1)
