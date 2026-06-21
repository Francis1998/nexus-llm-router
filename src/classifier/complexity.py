"""Lightweight complexity and domain classifiers."""

import math

from classifier.features import PromptFeatures
from router.schemas import DomainTag, LatencyRequirement


class LogisticComplexityClassifier:
    """Logistic regression classifier over deterministic prompt features."""

    def __init__(self) -> None:
        """Initialize calibrated feature weights."""
        self._bias = -1.55
        self._weights: dict[str, float] = {
            "word_count": 0.008,
            "question_count": 0.18,
            "code_hits": 0.32,
            "medical_hits": 0.26,
            "legal_hits": 0.24,
            "instruction_hits": 0.34,
        }

    def predict_score(self, features: PromptFeatures) -> float:
        """Predict prompt complexity on a 0 to 1 scale.

        Args:
            features: Extracted prompt features.

        Returns:
            Complexity score between 0 and 1.
        """
        linear_score = (
            self._bias
            + self._weights["word_count"] * min(features.word_count, 600)
            + self._weights["question_count"] * features.question_count
            + self._weights["code_hits"] * features.code_hits
            + self._weights["medical_hits"] * features.medical_hits
            + self._weights["legal_hits"] * features.legal_hits
            + self._weights["instruction_hits"] * features.instruction_hits
        )
        return 1.0 / (1.0 + math.exp(-linear_score))


class DomainClassifier:
    """Deterministic domain classifier for supported routing domains."""

    def classify(self, features: PromptFeatures) -> DomainTag:
        """Classify a prompt domain from extracted features.

        Args:
            features: Extracted prompt features.

        Returns:
            Domain tag for the prompt.
        """
        domain_scores = {
            DomainTag.CODE: features.code_hits,
            DomainTag.MEDICAL: features.medical_hits,
            DomainTag.LEGAL: features.legal_hits,
            DomainTag.GENERAL: 0,
        }
        chosen_domain = max(domain_scores, key=lambda domain: domain_scores[domain])
        return chosen_domain if domain_scores[chosen_domain] > 0 else DomainTag.GENERAL


class LatencyClassifier:
    """Infer latency requirements from prompt size and complexity."""

    def classify(self, features: PromptFeatures, complexity_score: float) -> LatencyRequirement:
        """Classify whether the request is realtime or batch.

        Args:
            features: Extracted prompt features.
            complexity_score: Complexity score between 0 and 1.

        Returns:
            Latency requirement for the request.
        """
        if features.word_count > 1200 or complexity_score > 0.88:
            return LatencyRequirement.BATCH
        return LatencyRequirement.REALTIME
