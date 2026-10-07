"""Unit tests for TokenizerVocabularyDriftAdvisor."""

from __future__ import annotations

import pytest

from safety.tokenizer_vocabulary_drift import TokenizerVocabularyDriftAdvisor


def test_within() -> None:
    """Good metric is within."""

    advice = TokenizerVocabularyDriftAdvisor().advise(request_id="r1", drift_ratio=0.02)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = TokenizerVocabularyDriftAdvisor().advise(request_id="r1", drift_ratio=0.1)
    assert advice.band == "soft"


def test_breach() -> None:
    """Bad metric is breach."""

    advice = TokenizerVocabularyDriftAdvisor().advise(request_id="r1", drift_ratio=0.25)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="drift_ratio"):
        TokenizerVocabularyDriftAdvisor().advise(request_id="r1", drift_ratio=-0.1)
