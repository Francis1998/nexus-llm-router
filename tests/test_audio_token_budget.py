"""Unit tests for AudioTokenBudgetAdvisor."""

from __future__ import annotations

import pytest

from safety.audio_token_budget import AudioTokenBudgetAdvisor


def test_within() -> None:
    """Band within."""

    advice = AudioTokenBudgetAdvisor().advise(request_id="r1", audio_token_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = AudioTokenBudgetAdvisor().advise(request_id="r1", audio_token_ratio=0.45)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = AudioTokenBudgetAdvisor().advise(request_id="r1", audio_token_ratio=0.9)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="audio_token_ratio"):
        AudioTokenBudgetAdvisor().advise(request_id="r1", audio_token_ratio=-0.1)
