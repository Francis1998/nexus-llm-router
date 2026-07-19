"""Regression tests for DomainClassifier tie-break priority."""

from classifier.complexity import DomainClassifier
from classifier.features import PromptFeatures
from router.schemas import DomainTag


def test_domain_classifier_tie_prefers_medical_over_code() -> None:
    """Equal medical and code hits must prefer MEDICAL, not CODE.

    ``max`` over a dict previously returned the first maximal key. Because
    ``CODE`` was inserted first, a prompt with equal medical and code keyword
    hits always classified as ``CODE``, skipping the higher-stakes medical
    safety path (Claude Sonnet 4.6). Ties now break MEDICAL > LEGAL > CODE >
    GENERAL.
    """
    features = PromptFeatures(
        character_count=40,
        word_count=6,
        question_count=0,
        code_hits=2,
        medical_hits=2,
        legal_hits=0,
        instruction_hits=0,
    )

    assert DomainClassifier().classify(features) is DomainTag.MEDICAL


def test_domain_classifier_tie_prefers_legal_over_code() -> None:
    """Equal legal and code hits must prefer LEGAL over CODE."""
    features = PromptFeatures(
        character_count=40,
        word_count=6,
        question_count=0,
        code_hits=1,
        medical_hits=0,
        legal_hits=1,
        instruction_hits=0,
    )

    assert DomainClassifier().classify(features) is DomainTag.LEGAL


def test_domain_classifier_still_picks_clear_majority() -> None:
    """A clear hit majority must still win regardless of tie-break priority."""
    features = PromptFeatures(
        character_count=40,
        word_count=6,
        question_count=0,
        code_hits=5,
        medical_hits=1,
        legal_hits=1,
        instruction_hits=0,
    )

    assert DomainClassifier().classify(features) is DomainTag.CODE
