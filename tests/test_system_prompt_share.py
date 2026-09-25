"""Unit tests for SystemPromptShareAdvisor."""

from __future__ import annotations

import pytest

from safety.system_prompt_share import SystemPromptShareAdvisor


def test_ok_band() -> None:
    """Small system prompt share is ok."""

    advice = SystemPromptShareAdvisor().advise(
        system_prompt_tokens=500,
        context_window_tokens=128000,
    )
    assert advice.band == "ok"


def test_heavy_band() -> None:
    """Share between 0.2 and 0.35 is heavy."""

    advice = SystemPromptShareAdvisor().advise(
        system_prompt_tokens=30000,
        context_window_tokens=100000,
    )
    assert advice.band == "heavy"


def test_bloated_band() -> None:
    """Share >= 0.35 is bloated."""

    advice = SystemPromptShareAdvisor().advise(
        system_prompt_tokens=40000,
        context_window_tokens=100000,
    )
    assert advice.band == "bloated"


def test_invalid_window_raises() -> None:
    """Non-positive context window raises ValueError."""

    with pytest.raises(ValueError, match="context_window_tokens"):
        SystemPromptShareAdvisor().advise(
            system_prompt_tokens=0,
            context_window_tokens=0,
        )
