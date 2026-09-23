"""Tests for PromptCompressionRatioAdvisor."""

from __future__ import annotations

import pytest

from safety.prompt_compression_ratio import PromptCompressionRatioAdvisor


def test_keep() -> None:
    advice = PromptCompressionRatioAdvisor().advise(prompt_tokens=1_000, context_window=8_000)
    assert advice.band == "keep"


def test_compress() -> None:
    advice = PromptCompressionRatioAdvisor().advise(prompt_tokens=5_000, context_window=8_000)
    assert advice.band == "compress"


def test_aggressive() -> None:
    advice = PromptCompressionRatioAdvisor().advise(prompt_tokens=7_500, context_window=8_000)
    assert advice.band == "aggressive"


def test_invalid() -> None:
    with pytest.raises(ValueError, match="context_window"):
        PromptCompressionRatioAdvisor().advise(prompt_tokens=10, context_window=0)
