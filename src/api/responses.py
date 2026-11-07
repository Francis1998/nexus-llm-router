"""OpenAI Responses API compatibility layer for ``POST /v1/responses``.

Translates the documented text-only subset of the Responses API into nexus
router messages, builds ``response`` objects from routed results, and
serializes the official streaming event sequence. Nexus is stateless, so
features that need stored responses or tool execution are rejected with an
OpenAI-style 400 instead of being silently ignored.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from time import time
from typing import Any, Literal
from uuid import uuid4

from api.errors import ROUTING_ERRORS, GatewayError, routing_error
from api.schemas import (
    NexusRoutingMetadata,
    ResponseCreateRequest,
    ResponseErrorInfo,
    ResponseObject,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from router.engine import RoutingFailedError
from router.schemas import ChatMessage, RouterResponse

RESPONSES_PATH = "/v1/responses"

_RouterRole = Literal["system", "user", "assistant"]
_ROLES: dict[str, _RouterRole] = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "developer": "system",
}
# Text-bearing content part types and the field holding their text. ``output_text``
# and ``refusal`` appear when clients replay earlier assistant output items.
_TEXT_PART_FIELDS = {"input_text": "text", "output_text": "text", "refusal": "refusal"}
_PASSIVE_TOOL_CHOICES = (None, "auto", "none")


def reject_unsupported(payload: ResponseCreateRequest) -> None:
    """Reject Responses API features that nexus cannot honor.

    Args:
        payload: Parsed request body.

    Raises:
        GatewayError: 400 ``unsupported_parameter`` naming the offending field.
    """
    text_format = (payload.text or {}).get("format")
    format_type = text_format.get("type") if isinstance(text_format, Mapping) else None
    stateless = "nexus is stateless; resend earlier turns in `input`"
    checks = (
        ("previous_response_id", payload.previous_response_id is not None, stateless),
        ("conversation", payload.conversation is not None, stateless),
        ("background", payload.background is True, "background mode needs stored responses"),
        ("store", payload.store is True, "responses are never stored; omit `store` or send false"),
        ("tools", bool(payload.tools), "tool calling is not routed by this endpoint"),
        (
            "tool_choice",
            payload.tool_choice not in _PASSIVE_TOOL_CHOICES,
            "tool calling is not routed by this endpoint",
        ),
        ("prompt", payload.prompt is not None, "send the rendered prompt text in `input`"),
        ("text.format", format_type not in (None, "text"), "only plain-text output is supported"),
    )
    for param, rejected, reason in checks:
        if rejected:
            raise GatewayError(
                400,
                f"Unsupported parameter: '{param}' is not supported by nexus /v1/responses "
                f"({reason}).",
                code="unsupported_parameter",
                param=param,
            )


def to_router_messages(payload: ResponseCreateRequest) -> list[ChatMessage]:
    """Translate ``instructions`` + ``input`` into router chat messages.

    ``instructions`` becomes a leading system message, a string ``input`` a user
    message, and ``developer`` input items map to the ``system`` role.

    Args:
        payload: Parsed request body.

    Returns:
        Router chat messages in conversation order.

    Raises:
        GatewayError: 400 when ``input`` is missing or contains non-text items.
    """
    messages: list[ChatMessage] = []
    if payload.instructions:
        messages.append(ChatMessage(role="system", content=payload.instructions))
    if payload.input is None:
        raise GatewayError(
            400,
            "Missing required parameter: 'input'.",
            code="missing_required_parameter",
            param="input",
        )
    if isinstance(payload.input, str):
        messages.append(ChatMessage(role="user", content=payload.input))
        return messages
    if not payload.input:
        raise GatewayError(
            400,
            "Invalid 'input': expected at least one input item.",
            code="empty_array",
            param="input",
        )
    for index, item in enumerate(payload.input):
        messages.append(_input_item_to_message(item, index))
    return messages


def _input_item_to_message(item: Mapping[str, Any], index: int) -> ChatMessage:
    """Convert one ``input`` list item into a router chat message."""
    item_type = item.get("type", "message")
    if item_type != "message":
        raise GatewayError(
            400,
            f"Unsupported input item type '{item_type}' at 'input[{index}]': nexus "
            "/v1/responses accepts message items with text content only.",
            code="unsupported_value",
            param=f"input[{index}].type",
        )
    role = _ROLES.get(str(item.get("role")))
    if role is None:
        raise GatewayError(
            400,
            f"Invalid value for 'input[{index}].role': expected one of 'user', "
            "'assistant', 'system', or 'developer'.",
            code="invalid_value",
            param=f"input[{index}].role",
        )
    return ChatMessage(role=role, content=_content_text(item.get("content"), index))


def _content_text(content: object, index: int) -> str:
    """Flatten message content (string or text parts) into router text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise GatewayError(
            400,
            f"Invalid type for 'input[{index}].content': expected a string or an array "
            "of content parts.",
            code="invalid_type",
            param=f"input[{index}].content",
        )
    segments: list[str] = []
    for part_index, part in enumerate(content):
        location = f"input[{index}].content[{part_index}]"
        part_type = part.get("type") if isinstance(part, Mapping) else None
        field = _TEXT_PART_FIELDS.get(str(part_type))
        if field is None:
            raise GatewayError(
                400,
                f"Unsupported content part type '{part_type}' at '{location}': nexus routes "
                "text only, so use 'input_text' parts (replayed 'output_text' and "
                "'refusal' parts are also accepted).",
                code="unsupported_value",
                param=f"{location}.type",
            )
        value = part.get(field)
        if not isinstance(value, str):
            raise GatewayError(
                400,
                f"Invalid type for '{location}.{field}': expected a string.",
                code="invalid_type",
                param=f"{location}.{field}",
            )
        segments.append(value)
    return "\n".join(segments)


def validation_error(errors: Sequence[Any]) -> GatewayError:
    """Convert FastAPI request-validation errors into an OpenAI-style 400.

    Args:
        errors: ``RequestValidationError.errors()`` entries.

    Returns:
        Gateway error naming the first invalid top-level parameter.
    """
    first = errors[0] if errors and isinstance(errors[0], Mapping) else {}
    if first.get("type") == "json_invalid":
        return GatewayError(400, "We could not parse the JSON body of your request.")
    location = [part for part in first.get("loc", ()) if part != "body"]
    param = str(location[0]) if location else None
    detail = str(first.get("msg", "invalid request"))
    if param is None:
        return GatewayError(400, f"Invalid request body: {detail}.", code="invalid_value")
    return GatewayError(
        400, f"Invalid value for '{param}': {detail}.", code="invalid_value", param=param
    )


def new_response_id() -> str:
    """Return a new ``resp_`` identifier."""
    return f"resp_{uuid4().hex}"


def draft_response(
    payload: ResponseCreateRequest, *, response_id: str, created_at: int, model: str
) -> ResponseObject:
    """Return the ``in_progress`` response that echoes the request settings.

    Args:
        payload: Parsed request body.
        response_id: ``resp_`` identifier shared by every lifecycle event.
        created_at: Unix timestamp (seconds) when the request was accepted.
        model: Model name to report until routing picks one.

    Returns:
        In-progress response object with empty output and no usage.
    """
    return ResponseObject(
        id=response_id,
        created_at=created_at,
        status="in_progress",
        model=model,
        instructions=payload.instructions,
        max_output_tokens=payload.max_output_tokens,
        parallel_tool_calls=(
            True if payload.parallel_tool_calls is None else payload.parallel_tool_calls
        ),
        temperature=1.0 if payload.temperature is None else payload.temperature,
        tool_choice=payload.tool_choice if isinstance(payload.tool_choice, str) else "auto",
        user=payload.user,
        safety_identifier=payload.safety_identifier,
        metadata=dict(payload.metadata or {}),
    )


def complete_response(
    draft: ResponseObject,
    router_response: RouterResponse,
    message_id: str,
    *,
    text: str | None = None,
) -> ResponseObject:
    """Return the ``completed`` response for a routed result.

    Args:
        draft: In-progress response from :func:`draft_response`.
        router_response: Routed completion.
        message_id: ``msg_`` identifier of the assistant output item.
        text: Output text override; streams pass the text actually delivered
            as deltas so the final snapshot matches what the client received.

    Returns:
        Completed response with output, usage, and nexus routing metadata.
    """
    output_text = router_response.content if text is None else text
    return draft.model_copy(
        update={
            "status": "completed",
            "completed_at": int(time()),
            "model": router_response.model_used,
            "output": [_output_message(message_id, output_text, "completed")],
            "usage": ResponseUsage(
                input_tokens=router_response.input_tokens,
                output_tokens=router_response.output_tokens,
                total_tokens=router_response.input_tokens + router_response.output_tokens,
            ),
            "nexus": NexusRoutingMetadata(
                routing_strategy=router_response.routing_strategy,
                rationale=router_response.rationale,
                cost_usd=router_response.cost_usd,
                latency_ms=router_response.latency_ms,
                request_id=router_response.request_id,
            ),
        }
    )


def _output_message(
    message_id: str, text: str, status: Literal["completed", "incomplete"]
) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=message_id, status=status, content=[ResponseOutputText(text=text)]
    )


class _EventWriter:
    """Serialize streaming events as SSE frames with increasing sequence numbers."""

    def __init__(self) -> None:
        self._sequence_number = 0

    def frame(self, event_type: str, **fields: object) -> str:
        payload = {"type": event_type, "sequence_number": self._sequence_number, **fields}
        self._sequence_number += 1
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


async def stream_response_events(
    draft: ResponseObject,
    first_chunk: tuple[str, str] | None,
    chunks: AsyncGenerator[tuple[str, str], None],
    completed: Sequence[RouterResponse],
) -> AsyncIterator[str]:
    """Yield the Responses API streaming event sequence for a routed stream.

    The caller pulls ``first_chunk`` before the HTTP response starts so routing
    failures that happen before any output surface as normal HTTP errors.
    Failures after output started end the stream with ``response.failed``.

    Args:
        draft: In-progress response from :func:`draft_response`.
        first_chunk: First ``(model, text)`` pair, or ``None`` for empty output.
        chunks: Remaining router stream.
        completed: Filled by ``NexusRouter.complete_stream(on_complete=...)``.

    Yields:
        ``event:``/``data:`` SSE frames.
    """
    writer = _EventWriter()
    message_id = f"msg_{uuid4().hex}"
    item = {"item_id": message_id, "output_index": 0, "content_index": 0}
    model = first_chunk[0] if first_chunk is not None else draft.model
    draft = draft.model_copy(update={"model": model})
    text = ""
    try:
        yield writer.frame("response.created", response=draft.model_dump(mode="json"))
        yield writer.frame("response.in_progress", response=draft.model_dump(mode="json"))
        in_progress_item = ResponseOutputMessage(id=message_id, status="in_progress")
        yield writer.frame(
            "response.output_item.added",
            output_index=0,
            item=in_progress_item.model_dump(mode="json"),
        )
        yield writer.frame(
            "response.content_part.added",
            **item,
            part=ResponseOutputText(text="").model_dump(mode="json"),
        )
        try:
            if first_chunk is not None and first_chunk[1]:
                text += first_chunk[1]
                yield writer.frame(
                    "response.output_text.delta", **item, delta=first_chunk[1], logprobs=[]
                )
            async for chunk_model, delta in chunks:
                model = chunk_model
                if delta:
                    text += delta
                    yield writer.frame(
                        "response.output_text.delta", **item, delta=delta, logprobs=[]
                    )
            if not completed:
                raise RoutingFailedError("stream ended without a routed response")
        except ROUTING_ERRORS as exception:
            error = routing_error(exception)
            failed = draft.model_copy(
                update={
                    "status": "failed",
                    "model": model,
                    "error": ResponseErrorInfo(
                        code="rate_limit_exceeded" if error.status_code == 429 else "server_error",
                        message=error.message,
                        type=error.error_type,
                    ),
                    "output": [_output_message(message_id, text, "incomplete")],
                }
            )
            yield writer.frame("response.failed", response=failed.model_dump(mode="json"))
            return
        part = ResponseOutputText(text=text)
        yield writer.frame("response.output_text.done", **item, text=text, logprobs=[])
        yield writer.frame("response.content_part.done", **item, part=part.model_dump(mode="json"))
        yield writer.frame(
            "response.output_item.done",
            output_index=0,
            item=_output_message(message_id, text, "completed").model_dump(mode="json"),
        )
        final = complete_response(draft, completed[-1], message_id, text=text)
        yield writer.frame("response.completed", response=final.model_dump(mode="json"))
    finally:
        await chunks.aclose()
