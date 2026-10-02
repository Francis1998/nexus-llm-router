"""Unit tests for TokenizerMismatchAdvisor."""

from __future__ import annotations

import pytest

from safety.tokenizer_mismatch import TokenizerMismatchAdvisor


def test_within() -> None:
    """Low mismatch is within."""

    advice = TokenizerMismatchAdvisor().advise(request_id="r1", mismatch_ratio=0.01)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid mismatch is soft."""

    advice = TokenizerMismatchAdvisor().advise(request_id="r1", mismatch_ratio=0.08)
    assert advice.band == "soft"


def test_breach() -> None:
    """High mismatch is breach."""

    advice = TokenizerMismatchAdvisor().advise(request_id="r1", mismatch_ratio=0.25)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative ratio raises."""

    with pytest.raises(ValueError, match="mismatch_ratio"):
        TokenizerMismatchAdvisor().advise(request_id="r1", mismatch_ratio=-0.1)
