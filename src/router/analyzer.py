"""Observe phase implementation for request analysis."""

from classifier.complexity import DomainClassifier, LatencyClassifier, LogisticComplexityClassifier
from classifier.features import extract_prompt_features
from router.schemas import RouterRequest, TaskSignals

CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_prompt_tokens(prompt_text: str) -> int:
    """Return the prompt-token estimate used for routing and budget checks.

    Nexus estimates about four characters per token, independent of the
    provider tokenizer, so the number is known before a model is chosen.

    Args:
        prompt_text: Concatenated prompt (``RouterRequest.prompt_text``).

    Returns:
        Estimated prompt tokens (at least 1).
    """
    return max(1, len(prompt_text) // CHARS_PER_TOKEN_ESTIMATE)


class RequestAnalyzer:
    """Extract routing signals from incoming requests."""

    def __init__(
        self,
        complexity_classifier: LogisticComplexityClassifier | None = None,
        domain_classifier: DomainClassifier | None = None,
        latency_classifier: LatencyClassifier | None = None,
    ) -> None:
        """Initialize analyzer dependencies.

        Args:
            complexity_classifier: Optional complexity classifier override.
            domain_classifier: Optional domain classifier override.
            latency_classifier: Optional latency classifier override.
        """
        self._complexity_classifier = complexity_classifier or LogisticComplexityClassifier()
        self._domain_classifier = domain_classifier or DomainClassifier()
        self._latency_classifier = latency_classifier or LatencyClassifier()

    def analyze(self, request: RouterRequest) -> TaskSignals:
        """Extract complexity, domain, latency, and token budget signals.

        Args:
            request: Router request.

        Returns:
            Task signals used by routing strategies.
        """
        features = extract_prompt_features(request.prompt_text)
        complexity_score = self._complexity_classifier.predict_score(features)
        domain_tag = self._domain_classifier.classify(features)
        latency_requirement = self._latency_classifier.classify(features, complexity_score)
        estimated_tokens = estimate_prompt_tokens(request.prompt_text)
        return TaskSignals(
            complexity_score=complexity_score,
            domain_tag=domain_tag,
            latency_requirement=latency_requirement,
            token_budget=request.token_budget,
            prompt_tokens_estimate=estimated_tokens,
        )
