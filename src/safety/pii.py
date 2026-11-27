"""PII scrubbing controls."""

import json
import re
from typing import Protocol

from router.schemas import ChatMessage, ToolCall

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
# A leading ``\b`` cannot match before a ``(`` or ``+`` (non-word characters), so
# numbers written as ``(415) 555-1234`` or ``+1 415 555 1234`` had their leading
# ``(``/``+`` left dangling outside the redaction. Use non-consuming
# ``(?<!\w)``/``(?!\w)`` boundaries that hold whether the number starts with a
# digit, a ``(``, or a ``+`` so the whole number is replaced.
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)")


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
        """Scrub message text, tool results, and tool-call arguments.

        Tool results (``tool`` message content) and tool-call arguments go
        through :meth:`scrub_json_text`, so JSON stays valid JSON. Tool-call ids
        and function names are kept, so calls still pair with their results.

        Args:
            messages: Original chat messages.

        Returns:
            Scrubbed chat messages.
        """
        if not self._enabled:
            return messages
        scrubbed: list[ChatMessage] = []
        for message in messages:
            update: dict[str, object] = {
                "content": (
                    self.scrub_json_text(message.content)
                    if message.role == "tool"
                    else self.scrub_text(message.content)
                )
            }
            if message.tool_calls:
                update["tool_calls"] = [self._scrub_tool_call(call) for call in message.tool_calls]
            scrubbed.append(message.model_copy(update=update))
        return scrubbed

    def scrub_json_text(self, text: str) -> str:
        """Scrub a JSON document leaf by leaf, or ``text`` as plain text when it is not JSON.

        Redacting serialized JSON as text could break it (a phone-length number
        would become a bare ``[REDACTED_PHONE]``), so every string, object key,
        and number is scrubbed on its own; a number that would be redacted is
        replaced by the redaction string, and keys that redact alike get distinct
        suffixed names so no entry is lost. Documents without PII are returned
        verbatim.

        Args:
            text: Tool-call arguments or a tool result.

        Returns:
            Redacted text (valid JSON whenever ``text`` was).
        """
        try:
            decoded = json.loads(text)
        except ValueError:
            return self.scrub_text(text)
        scrubbed = self._scrub_json(decoded)
        if scrubbed == decoded:
            return text
        return json.dumps(scrubbed, ensure_ascii=False)

    def _scrub_tool_call(self, call: ToolCall) -> ToolCall:
        """Return ``call`` with its JSON arguments scrubbed."""
        function = call.function.model_copy(
            update={"arguments": self.scrub_json_text(call.function.arguments)}
        )
        return call.model_copy(update={"function": function})

    def _scrub_json(self, value: object) -> object:
        """Return a decoded JSON ``value`` with every string, key, and number scrubbed."""
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int | float):
            literal = json.dumps(value)
            redacted = self.scrub_text(literal)
            return value if redacted == literal else redacted
        if isinstance(value, list):
            return [self._scrub_json(item) for item in value]
        if isinstance(value, dict):
            return self._scrub_object(value)
        return value

    def _scrub_object(self, value: dict[object, object]) -> dict[str, object]:
        """Scrub a decoded JSON object's keys and values without dropping an entry.

        Keys that redact to the same text (two email addresses both become
        ``[REDACTED_EMAIL]``) get ``_2``, ``_3``, ... suffixes. Keys without PII keep
        their names, and a suffixed name never replaces one of them.
        """
        names = [str(key) for key in value]
        redacted = [self.scrub_text(name) for name in names]
        used = {name for name, new in zip(names, redacted, strict=True) if name == new}
        scrubbed: dict[str, object] = {}
        for name, new, item in zip(names, redacted, value.values(), strict=True):
            key = new
            if new != name:
                suffix = 2
                while key in used:
                    key = f"{new}_{suffix}"
                    suffix += 1
                used.add(key)
            scrubbed[key] = self._scrub_json(item)
        return scrubbed

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
