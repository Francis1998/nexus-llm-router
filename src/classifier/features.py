"""Prompt feature extraction utilities."""

import re
from dataclasses import dataclass

# Each keyword is bounded by ``\b`` on both sides rather than a trailing literal
# space. Requiring a following space silently missed keywords immediately
# followed by a newline, colon, or parenthesis — for example an idiomatic SQL
# query beginning ``SELECT\n`` or a Python ``class Foo:`` — which are unmistakably
# code. ``SELECT`` stays upper-case only (matching the deliberately
# case-sensitive intent), and the trailing ``\b`` still excludes substrings such
# as ``subclass`` or ``important``.
CODE_PATTERN = re.compile(
    r"```|\bdef\b|\bclass\b|\bimport\b|\bSELECT\b|\bfunction\b|\bconst\b|\basync\b"
)
MEDICAL_PATTERN = re.compile(r"\b(patient|diagnosis|clinical|medical|symptom|treatment)\b", re.I)
LEGAL_PATTERN = re.compile(r"\b(contract|clause|statute|liability|legal|compliance)\b", re.I)
# Instruction verbs plus their inflections, matched with explicit inflectional
# endings rather than a permissive trailing ``\w*``. The ``\w*`` form both
# over-matched unrelated words that merely *start* with a verb (``prove`` ->
# ``proverb``, ``design`` -> ``designated``/``designation``) and, because it
# could not model the silent-``e`` drop, *under*-matched genuine ``-ing`` forms
# (``analyzing``/``optimizing``/``proving``/``comparing`` were missed). Listing
# the real endings (with British ``-ise``/``-yse`` spellings and the
# consonant-doubling ``debugged``/``debugging``) fixes both, and the trailing
# ``\b`` keeps unrelated longer words out.
INSTRUCTION_PATTERN = re.compile(
    r"\b(?:"
    r"analy[sz]e[sd]?|analy[sz]ing|"
    r"debug(?:s|ged|ging)?|"
    r"prove[nsd]?|proving|"
    r"design(?:s|ed|ing)?|"
    r"optimi[sz]e[sd]?|optimi[sz]ing|"
    r"compare[sd]?|comparing"
    r")\b",
    re.I,
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
