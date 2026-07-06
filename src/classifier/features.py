"""Prompt feature extraction utilities."""

import re
from dataclasses import dataclass

CODE_PATTERN = re.compile(
    r"```|\bdef |\bclass |\bimport |\bSELECT |\bfunction |\bconst |\basync "
)
MEDICAL_PATTERN = re.compile(r"\b(patient|diagnosis|clinical|medical|symptom|treatment)\b", re.I)
LEGAL_PATTERN = re.compile(r"\b(contract|clause|statute|liability|legal|compliance)\b", re.I)
INSTRUCTION_PATTERN = re.compile(
    r"\b(analyze|debug|prove|design|optimize|compare)\w*", re.I
)


@dataclass(frozen=True)
class PromptFeatures:
    """Numeric prompt features used by routing classifiers."""

    character_count: int
    word_count: int
    question_count: int
    code_hits: int
    medical_hits: int
    legal_hits: int
    instruction_hits: int


def extract_prompt_features(prompt_text: str) -> PromptFeatures:
    """Extract deterministic prompt features.

    Args:
        prompt_text: User prompt text.

    Returns:
        Feature vector derived from the prompt.
    """
    words = re.findall(r"\w+", prompt_text)
    instruction_hits = len(INSTRUCTION_PATTERN.findall(prompt_text))
    return PromptFeatures(
        character_count=len(prompt_text),
        word_count=len(words),
        question_count=prompt_text.count("?"),
        code_hits=len(CODE_PATTERN.findall(prompt_text)),
        medical_hits=len(MEDICAL_PATTERN.findall(prompt_text)),
        legal_hits=len(LEGAL_PATTERN.findall(prompt_text)),
        instruction_hits=instruction_hits,
    )
