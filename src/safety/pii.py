"""PII scrubbing controls."""

import re
from typing import Protocol

from router.schemas import ChatMessage

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


class PresidioAnalyzerProtocol(Protocol):
    """Minimal protocol for Presidio analyzer engines."""

    def analyze(self, text: str, language: str) -> list[object]:
        """Analyze text for PII entities."""


class PresidioAnonymizerProtocol(Protocol):
    """Minimal protocol for Presidio anonymizer engines."""

    def anonymize(self, text: str, analyzer_results: list[object]) -> object:
        """Anonymize text from analyzer results."""


class PiiScrubber:
    """Scrub PII from chat messages before provider dispatch."""

    def __init__(
        self,
        enabled: bool,
        presidio_analyzer: PresidioAnalyzerProtocol | None = None,
        presidio_anonymizer: PresidioAnonymizerProtocol | None = None,
    ) -> None:
        """Initialize scrubber dependencies.

        Args:
            enabled: Whether scrubbing should run.
            presidio_analyzer: Optional Presidio analyzer engine.
            presidio_anonymizer: Optional Presidio anonymizer engine.
        """
        self._enabled = enabled
        self._presidio_analyzer = presidio_analyzer
        self._presidio_anonymizer = presidio_anonymizer

    def scrub_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Scrub all message contents.

        Args:
            messages: Original chat messages.

        Returns:
            Scrubbed chat messages.
        """
        if not self._enabled:
            return messages
        return [
            ChatMessage(role=message.role, content=self.scrub_text(message.content))
            for message in messages
        ]

    def scrub_text(self, text: str) -> str:
        """Scrub PII from one text field.

        Args:
            text: Input text.

        Returns:
            Redacted text.
        """
        redacted_text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
        redacted_text = PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted_text)
        if self._presidio_analyzer is None or self._presidio_anonymizer is None:
            return redacted_text
        analyzer_results = self._presidio_analyzer.analyze(text=redacted_text, language="en")
        anonymized = self._presidio_anonymizer.anonymize(
            text=redacted_text,
            analyzer_results=analyzer_results,
        )
        anonymized_text = getattr(anonymized, "text", redacted_text)
        return anonymized_text if isinstance(anonymized_text, str) else redacted_text
