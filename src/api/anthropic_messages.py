"""Anthropic Messages API compatibility layer for ``POST /v1/messages``.

Translates the documented text-only subset of the Anthropic Messages API into
nexus router messages, renders routed results as Anthropic ``message`` objects,
and serializes Anthropic's streaming event sequence. Provider adapters are
text-only, so tools, extended thinking, structured outputs, and non-text
content blocks are rejected with an Anthropic-style 400 instead of being
silently dropped. ``stop_sequences`` are enforced at the gateway because
adapters cannot ask providers to stop at custom strings.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import Any
from uuid import uuid4

from api.errors import ROUTING_ERRORS, GatewayError, routing_error
from api.schemas import (
    AnthropicMessage,
    AnthropicStopReason,
    AnthropicTextBlock,
    AnthropicUsage,
    MessageCountTokensRequest,
    MessageCreateRequest,
    NexusRoutingMetadata,
)
from router.engine import RoutingFailedError
from router.schemas import ChatMessage, FinishReason, RouterResponse

MESSAGES_PATH = "/v1/messages"
COUNT_TOKENS_PATH = "/v1/messages/count_tokens"

_PASSIVE_TOOL_CHOICES = ("auto", "none")
_TEXT_BLOCK_FIELDS = frozenset({"type", "text", "cache_control", "citations"})
_OUTPUT_CONFIG_FIELDS = frozenset({"effort", "format"})
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
_THINKING_BLOCK_TYPES = ("thinking", "redacted_thinking")


def is_messages_path(path: str) -> bool:
    """Return whether ``path`` belongs to the Anthropic Messages API surface.

    Args:
        path: Request URL path.

    Returns:
        ``True`` for ``/v1/messages`` and every path below it.
    """
    return path == MESSAGES_PATH or path.startswith(f"{MESSAGES_PATH}/")


def _invalid(message: str) -> GatewayError:
    return GatewayError(400, message)


def reject_unsupported(payload: MessageCountTokensRequest) -> None:
    """Reject Messages API features nexus cannot honor.

    Args:
        payload: Parsed ``/v1/messages`` or ``/v1/messages/count_tokens`` body.

    Raises:
        GatewayError: 400 whose message starts with the offending field.
    """
    if payload.tools:
        raise _invalid(
            "tools: tool use is not supported by nexus /v1/messages (provider adapters are "
            "text-only); remove `tools`."
        )
    if payload.tool_choice is not None and (
        payload.tool_choice.get("type") not in _PASSIVE_TOOL_CHOICES
    ):
        raise _invalid(
            'tool_choice: only {"type": "auto"} or {"type": "none"} is accepted because '
            "nexus /v1/messages does not route tool use."
        )
    if payload.thinking is not None and payload.thinking.get("type") != "disabled":
        raise _invalid(
            "thinking: extended thinking is not supported by nexus /v1/messages; omit "
            '`thinking` or send {"type": "disabled"}.'
        )
    _check_output_config(payload.output_config)
    if isinstance(payload, MessageCreateRequest):
        _reject_unsupported_create(payload)


def _check_output_config(output_config: Mapping[str, Any] | None) -> None:
    if output_config is None:
        return
    for key in output_config:
        if key not in _OUTPUT_CONFIG_FIELDS:
            raise _invalid(f"output_config.{key}: Extra inputs are not permitted")
    if output_config.get("format") is not None:
        raise _invalid(
            "output_config.format: structured outputs are not supported by nexus "
            "/v1/messages; only plain-text output is routed."
        )
    effort = output_config.get("effort")
    if effort is not None and effort not in _EFFORT_LEVELS:
        raise _invalid(
            "output_config.effort: Input should be 'low', 'medium', 'high', 'xhigh' or 'max'"
        )


def _reject_unsupported_create(payload: MessageCreateRequest) -> None:
    if payload.container is not None:
        raise _invalid(
            "container: containers (code execution, skills) are not supported by nexus "
            "/v1/messages."
        )
    if payload.inference_geo is not None:
        raise _invalid(
            "inference_geo: nexus cannot pin where a routed provider runs inference; omit "
            "`inference_geo`."
        )
    for index, sequence in enumerate(payload.stop_sequences or ()):
        if not sequence.strip():
            raise _invalid(
                f"stop_sequences.{index}: each stop sequence must contain non-whitespace"
            )


def to_router_messages(payload: MessageCountTokensRequest) -> list[ChatMessage]:
    """Translate ``system`` + ``messages`` into router chat messages.

    ``system`` becomes a leading system message. Text blocks of one message (or
    of a block-list ``system``) are joined with newlines; ``cache_control`` and
    ``citations`` on text blocks are accepted and ignored.

    Args:
        payload: Parsed request body.

    Returns:
        Router chat messages in conversation order.

    Raises:
        GatewayError: 400 for empty or non-text content, or when the last
            message is an assistant turn (prefill is not supported). An empty
            ``messages`` list is rejected even with a ``system`` prompt, the
            Messages-format equivalent of the empty or system-only conversations
            ``api.errors.conversation_error`` rejects on the OpenAI endpoints.
    """
    messages: list[ChatMessage] = []
    system = _system_text(payload.system)
    if system.strip():
        messages.append(ChatMessage(role="system", content=system))
    if not payload.messages:
        raise _invalid("messages: at least one message is required")
    for index, message in enumerate(payload.messages):
        content = _content_text(message.content, f"messages.{index}.content")
        messages.append(ChatMessage(role=message.role, content=content))
    if payload.messages[-1].role != "user":
        last = len(payload.messages) - 1
        raise _invalid(
            f"messages.{last}.role: the final message must have role 'user'; nexus "
            "/v1/messages does not support assistant prefill (continuing a partial "
            "assistant turn)."
        )
    return messages


def _system_text(system: object) -> str:
    if system is None or isinstance(system, str):
        return system or ""
    if not isinstance(system, list):
        raise _invalid("system: Input should be a valid string or a list of text blocks")
    return "\n".join(_block_text(block, f"system.{index}") for index, block in enumerate(system))


def _content_text(content: object, location: str) -> str:
    if isinstance(content, str):
        if not content.strip():
            raise _invalid(f"{location}: all messages must have non-empty content")
        return content
    if not isinstance(content, list):
        raise _invalid(f"{location}: Input should be a valid string or a list of content blocks")
    if not content:
        raise _invalid(f"{location}: all messages must have non-empty content")
    return "\n".join(
        _block_text(block, f"{location}.{index}") for index, block in enumerate(content)
    )


def _block_text(block: object, location: str) -> str:
    """Return the text of one ``text`` block; any other block type is a 400."""
    if not isinstance(block, Mapping):
        raise _invalid(f"{location}: Input should be a valid content block object")
    block_type = block.get("type")
    if block_type is None:
        raise _invalid(f"{location}.type: Field required")
    if block_type != "text":
        if block_type in _THINKING_BLOCK_TYPES:
            reason = "nexus does not return or replay extended thinking"
        elif isinstance(block_type, str) and "tool" in block_type:
            reason = "tool use is not routed by nexus"
        else:
            reason = "nexus routes text only"
        raise _invalid(
            f"{location}.type: '{block_type}' content blocks are not supported by nexus "
            f"/v1/messages ({reason}); send 'text' blocks."
        )
    for key in block:
        if key not in _TEXT_BLOCK_FIELDS:
            raise _invalid(f"{location}.{key}: Extra inputs are not permitted")
    text = block.get("text")
    if not isinstance(text, str):
        raise _invalid(f"{location}.text: Input should be a valid string")
    if not text.strip():
        raise _invalid(f"{location}.text: text content blocks must contain non-whitespace text")
    return text


def validation_error(errors: Sequence[Any]) -> GatewayError:
    """Convert FastAPI request-validation errors into an Anthropic-style 400.

    Args:
        errors: ``RequestValidationError.errors()`` entries.

    Returns:
        Gateway error whose message starts with the dotted path of the first
        invalid field, like the Anthropic API (``max_tokens: Field required``).
    """
    first = errors[0] if errors and isinstance(errors[0], Mapping) else {}
    if first.get("type") == "json_invalid":
        return _invalid("The request body is not valid JSON.")
    location = list(first.get("loc", ()))
    if location[:1] == ["body"]:
        location = location[1:]
    detail = str(first.get("msg", "invalid request"))
    if not location:
        return _invalid(f"Invalid request body: {detail}")
    return _invalid(f"{'.'.join(str(part) for part in location)}: {detail}")


def _find_stop_sequence(text: str, sequences: Sequence[str], start: int) -> tuple[int, str] | None:
    """Return ``(start index, sequence)`` of the stop sequence that ends first."""
    best: tuple[int, int, str] | None = None
    for sequence in sequences:
        index = text.find(sequence, start)
        if index >= 0 and (best is None or (index + len(sequence), index) < best[:2]):
            best = (index + len(sequence), index, sequence)
    return None if best is None else (best[1], best[2])


class StopSequenceFilter:
    """Enforce ``stop_sequences`` at the gateway for whole texts and streams.

    Adapters cannot ask a provider to stop at custom strings, so nexus cuts the
    output before the stop sequence that the model generated first (the one
    that ends first, as when generation stops). Streamed text is held back by
    ``len(longest stop sequence) - 1`` characters until it can no longer start
    a stop sequence, so clients never receive text that is cut later.
    """

    def __init__(self, stop_sequences: Sequence[str]) -> None:
        """Initialize the filter.

        Args:
            stop_sequences: Client ``stop_sequences`` (may be empty).
        """
        self._sequences = tuple(stop_sequences)
        self._holdback = max((len(sequence) for sequence in self._sequences), default=1) - 1
        self._text = ""
        self._sent = 0
        self.matched: str | None = None

    def feed(self, delta: str) -> str:
        """Add generated text and return the part that is now safe to emit.

        Args:
            delta: Next chunk of generated text.

        Returns:
            Text to emit; empty once a stop sequence matched.
        """
        if self.matched is not None:
            return ""
        self._text += delta
        match = _find_stop_sequence(self._text, self._sequences, self._sent)
        if match is None:
            end = len(self._text) - self._holdback
        else:
            end, self.matched = match
        if end <= self._sent:
            return ""
        visible = self._text[self._sent : end]
        self._sent = end
        return visible

    def flush(self) -> str:
        """Return the held-back text once generation ended without a match.

        Returns:
            Remaining text to emit (empty after a match).
        """
        if self.matched is not None:
            return ""
        visible = self._text[self._sent :]
        self._sent = len(self._text)
        return visible


def apply_stop_sequences(text: str, stop_sequences: Sequence[str]) -> tuple[str, str | None]:
    """Cut a complete output at its first stop sequence.

    Args:
        text: Full routed output.
        stop_sequences: Client ``stop_sequences``.

    Returns:
        ``(visible text, matched stop sequence or None)``.
    """
    stopper = StopSequenceFilter(stop_sequences)
    visible = stopper.feed(text) + stopper.flush()
    return visible, stopper.matched


def stop_reason(finish_reason: FinishReason, matched: str | None) -> AnthropicStopReason:
    """Map the router finish reason (and a gateway stop-sequence match) to Anthropic.

    Args:
        finish_reason: Normalized provider stop reason.
        matched: Stop sequence that cut the output, if any.

    Returns:
        ``stop_sequence``, ``max_tokens`` (provider hit the cap), or ``end_turn``.
    """
    if matched is not None:
        return "stop_sequence"
    return "max_tokens" if finish_reason == "length" else "end_turn"


def new_message_id() -> str:
    """Return a new ``msg_`` identifier."""
    return f"msg_{uuid4().hex}"


def complete_message(
    message_id: str, router_response: RouterResponse, stop_sequences: Sequence[str]
) -> AnthropicMessage:
    """Render a routed completion as an Anthropic ``message``.

    Args:
        message_id: ``msg_`` identifier.
        router_response: Routed completion (possibly from the response cache).
        stop_sequences: Client ``stop_sequences`` enforced on the output.

    Returns:
        Message with one text block, provider-reported usage, and the ``nexus``
        routing block.
    """
    text, matched = apply_stop_sequences(router_response.content, stop_sequences)
    return AnthropicMessage(
        id=message_id,
        model=router_response.model_used,
        content=[AnthropicTextBlock(text=text)],
        stop_reason=stop_reason(router_response.finish_reason, matched),
        stop_sequence=matched,
        usage=AnthropicUsage(
            input_tokens=router_response.input_tokens,
            output_tokens=router_response.output_tokens,
        ),
        nexus=NexusRoutingMetadata.from_router_response(router_response),
    )


def _sse(event: str, payload: Mapping[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _text_delta(text: str) -> str:
    delta = {"type": "text_delta", "text": text}
    return _sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta})


async def stream_message_events(
    *,
    message_id: str,
    model: str,
    input_tokens: int,
    first_chunk: tuple[str, str] | None,
    chunks: AsyncGenerator[tuple[str, str], None],
    completed: Sequence[RouterResponse],
    stop_sequences: Sequence[str],
) -> AsyncIterator[str]:
    """Yield Anthropic's streaming event sequence for a routed stream.

    ``message_start`` → ``content_block_start`` → ``ping`` →
    ``content_block_delta``\\* → ``content_block_stop`` → ``message_delta`` →
    ``message_stop``. The caller pulls chunks until the first non-empty one
    before the HTTP response starts, so failures before any output stay real
    HTTP errors; a failure after that ends the stream with one ``error`` event.

    Args:
        message_id: ``msg_`` identifier.
        model: Requested model, reported only if routing never named one.
        input_tokens: Router prompt estimate for ``message_start``; the final
            ``message_delta`` carries the provider-reported count.
        first_chunk: First non-empty ``(model, text)`` pair, an empty-text pair
            when the stream finished without text, or ``None``.
        chunks: Remaining router stream; drained fully even after a stop
            sequence matched so usage and spend are recorded.
        completed: Filled by ``NexusRouter.complete_stream(on_complete=...)``.
        stop_sequences: Client ``stop_sequences`` enforced on the stream.

    Yields:
        ``event:``/``data:`` SSE frames.
    """
    if completed:
        model = completed[-1].model_used
    elif first_chunk is not None:
        model = first_chunk[0]
    stopper = StopSequenceFilter(stop_sequences)
    start = AnthropicMessage(
        id=message_id, model=model, usage=AnthropicUsage(input_tokens=input_tokens, output_tokens=0)
    )
    try:
        message = start.model_dump(mode="json", exclude={"nexus"})
        yield _sse("message_start", {"type": "message_start", "message": message})
        block = AnthropicTextBlock(text="").model_dump(mode="json")
        yield _sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": block},
        )
        yield _sse("ping", {"type": "ping"})
        try:
            if first_chunk is not None and (text := stopper.feed(first_chunk[1])):
                yield _text_delta(text)
            async for _, delta in chunks:
                if text := stopper.feed(delta):
                    yield _text_delta(text)
            if not completed:
                raise RoutingFailedError("stream ended without a routed response")
        except ROUTING_ERRORS as exception:
            yield _sse("error", routing_error(exception).to_anthropic_body())
            return
        if text := stopper.flush():
            yield _text_delta(text)
        final = completed[-1]
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason(final.finish_reason, stopper.matched),
                    "stop_sequence": stopper.matched,
                },
                "usage": {"input_tokens": final.input_tokens, "output_tokens": final.output_tokens},
                "nexus": NexusRoutingMetadata.from_router_response(final).model_dump(mode="json"),
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
    finally:
        await chunks.aclose()
