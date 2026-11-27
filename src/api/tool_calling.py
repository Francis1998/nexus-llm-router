"""Tool calling on ``POST /v1/chat/completions``: request checks and streamed tool calls.

Nexus forwards function tools to the routed model and returns the model's tool
calls to the client; it never executes a tool. Every tool-calling request
error is an OpenAI-style 400 (``{"error": {...}}``) naming the offending
parameter; error messages name indices, ids, and function names, never tool
arguments or results.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from time import time
from typing import Any
from uuid import uuid4

from api.errors import TOOL_CALLING_UNSUPPORTED, GatewayError
from api.schemas import ChatCompletionRequest
from router.schemas import TOOL_NAME_PATTERN, ChatMessage, NamedToolChoice, RouterResponse

__all__ = [
    "TOOL_CALLING_UNSUPPORTED",
    "ToolCallingRequestError",
    "tool_call_stream_frames",
    "tool_message_error",
    "tool_request_error",
    "tool_validation_error",
]

_TOOL_FIELDS = frozenset(
    {"tools", "tool_choice", "parallel_tool_calls", "functions", "function_call"}
)
_TOOL_MESSAGE_FIELDS = frozenset({"tool_calls", "tool_call_id"})
_TOOL_NAME = re.compile(TOOL_NAME_PATTERN)


class ToolCallingRequestError(Exception):
    """A tool-calling request failure answered with an OpenAI ``{"error": {...}}`` envelope."""

    def __init__(self, error: GatewayError) -> None:
        """Wrap the gateway error to render.

        Args:
            error: Status, message, and OpenAI error fields.
        """
        super().__init__(error.message)
        self.error = error


def tool_validation_error(errors: Sequence[Any]) -> GatewayError | None:
    """Return an OpenAI-style 400 for the first request-validation error in a tool field.

    Tool fields are ``tools``, ``tool_choice``, ``parallel_tool_calls``, the
    deprecated ``functions`` / ``function_call``, and the ``tool_calls`` /
    ``tool_call_id`` fields of ``messages``. Other validation errors keep the
    route's FastAPI 422.

    Args:
        errors: ``RequestValidationError.errors()`` entries.

    Returns:
        The error to return, or ``None`` when no error concerns a tool field.
    """
    for entry in errors:
        if not isinstance(entry, Mapping):
            continue
        location = [part for part in entry.get("loc", ()) if part != "body"]
        if not _is_tool_location(location):
            continue
        param = _param_path(location)
        detail = str(entry.get("msg", "invalid value")).removeprefix("Value error, ")
        return GatewayError(
            400, f"Invalid value for '{param}': {detail}.", code="invalid_value", param=param
        )
    return None


def _is_tool_location(location: Sequence[object]) -> bool:
    """Return whether a validation-error location lies in a tool-calling field."""
    if not location:
        return False
    if location[0] in _TOOL_FIELDS:
        return True
    return location[0] == "messages" and len(location) > 2 and location[2] in _TOOL_MESSAGE_FIELDS


def _param_path(location: Sequence[object]) -> str:
    """Render ``("tools", 0, "function", "name")`` as ``tools[0].function.name``.

    ``tool_choice`` errors stop at the parameter: their locations continue with
    pydantic's union-member tags, which are not request fields.
    """
    if location[0] == "tool_choice":
        return "tool_choice"
    path = ""
    for part in location:
        path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
    return path


def tool_request_error(payload: ChatCompletionRequest) -> GatewayError | None:
    """Return a 400 for a tool-calling request no provider could answer consistently.

    Checks the deprecated ``functions`` / ``function_call`` parameters, duplicate
    tool names, a ``tool_choice`` that needs tools the request lacks, and the
    pairing of tool calls and tool results in ``messages``
    (:func:`tool_message_error`).

    Args:
        payload: Parsed chat completion request.

    Returns:
        The OpenAI-style error to return, or ``None`` when the request is valid.
    """
    for legacy in ("functions", "function_call"):
        if getattr(payload, legacy) is not None:
            return GatewayError(
                400,
                f"Unsupported parameter: '{legacy}' is deprecated and not supported by nexus; "
                "send 'tools' and 'tool_choice' instead.",
                code="unsupported_parameter",
                param=legacy,
            )
    names = [tool.function.name for tool in payload.tools or ()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        return GatewayError(
            400,
            f"Invalid 'tools': function names must be unique; duplicated: {', '.join(duplicates)}.",
            code="invalid_value",
            param="tools",
        )
    choice = payload.tool_choice
    if isinstance(choice, NamedToolChoice) and choice.function.name not in names:
        return GatewayError(
            400,
            f"Invalid value for 'tool_choice': function '{choice.function.name}' is not "
            "declared in 'tools'.",
            code="invalid_value",
            param="tool_choice",
        )
    if choice == "required" and not names:
        return GatewayError(
            400,
            "Invalid value for 'tool_choice': 'required' needs at least one tool in 'tools'.",
            code="invalid_value",
            param="tool_choice",
        )
    return tool_message_error(payload.messages)


def tool_message_error(messages: Sequence[ChatMessage]) -> GatewayError | None:
    """Return a 400 when tool calls and tool results in ``messages`` do not pair up.

    Like OpenAI, every ``tool`` message must answer a call of the nearest
    preceding assistant ``tool_calls`` message (with only ``tool`` messages in
    between), and every call must be answered once before the conversation
    moves on. Anthropic and Gemini reject unpaired calls and results, so
    checking here keeps the outcome independent of the routed provider.
    Replayed calls need unique non-empty ids, valid function names, and
    arguments that are JSON objects.

    Args:
        messages: Chat messages of the request.

    Returns:
        The OpenAI-style error to return, or ``None`` when every call pairs up.
    """
    open_calls: dict[str, int] = {}
    answered: set[str] = set()
    for index, message in enumerate(messages):
        param = f"messages[{index}]"
        if message.tool_call_id is not None and message.role != "tool":
            return _invalid(
                f"{param}.tool_call_id", "only messages with role 'tool' can carry tool_call_id"
            )
        if message.tool_calls:
            if message.role != "assistant":
                return _invalid(
                    f"{param}.tool_calls", "only assistant messages can carry tool_calls"
                )
            if error := _unanswered_error(open_calls, answered, index):
                return error
            if error := _tool_calls_error(message, param):
                return error
            open_calls = {call.id: index for call in message.tool_calls}
            answered = set()
            continue
        if message.role == "tool":
            if message.tool_call_id is None:
                return GatewayError(
                    400,
                    f"Missing required parameter: '{param}.tool_call_id'; messages with role "
                    "'tool' must answer a tool call.",
                    code="missing_required_parameter",
                    param=f"{param}.tool_call_id",
                )
            if message.tool_call_id not in open_calls:
                return _invalid(
                    f"{param}.tool_call_id",
                    "messages with role 'tool' must answer a tool call of the preceding "
                    f"assistant message; no such call has id '{message.tool_call_id}'",
                )
            if message.tool_call_id in answered:
                return _invalid(
                    f"{param}.tool_call_id",
                    f"tool call '{message.tool_call_id}' is already answered",
                )
            answered.add(message.tool_call_id)
            continue
        if error := _unanswered_error(open_calls, answered, index):
            return error
        open_calls, answered = {}, set()
    return _unanswered_error(open_calls, answered, len(messages))


def _unanswered_error(
    open_calls: dict[str, int], answered: set[str], index: int
) -> GatewayError | None:
    """Return a 400 when the latest assistant tool calls still lack results at ``index``."""
    missing = [call_id for call_id in open_calls if call_id not in answered]
    if not missing:
        return None
    origin = next(iter(open_calls.values()))
    return _invalid(
        f"messages[{origin}].tool_calls",
        "an assistant message with 'tool_calls' must be followed by 'tool' messages answering "
        f"each call; no result for: {', '.join(missing)}",
    )


def _tool_calls_error(message: ChatMessage, param: str) -> GatewayError | None:
    """Validate the ids, names, and arguments of one assistant message's tool calls."""
    seen: set[str] = set()
    for position, call in enumerate(message.tool_calls or ()):
        call_param = f"{param}.tool_calls[{position}]"
        if not call.id:
            return _invalid(f"{call_param}.id", "tool call ids must be non-empty")
        if call.id in seen:
            return _invalid(f"{call_param}.id", f"duplicate tool call id '{call.id}'")
        seen.add(call.id)
        if not _TOOL_NAME.match(call.function.name):
            return _invalid(
                f"{call_param}.function.name", f"function names must match {TOOL_NAME_PATTERN}"
            )
        try:
            arguments = json.loads(call.function.arguments)
        except ValueError:
            arguments = None
        if not isinstance(arguments, dict):
            return _invalid(
                f"{call_param}.function.arguments", "arguments must be a JSON-encoded object"
            )
    return None


def _invalid(param: str, reason: str) -> GatewayError:
    """Return an ``invalid_value`` 400 for ``param``."""
    return GatewayError(400, f"Invalid '{param}': {reason}.", code="invalid_value", param=param)


def tool_call_stream_frames(response: RouterResponse) -> list[str]:
    """Render a finished tool-calling completion as OpenAI ``chat.completion.chunk`` SSE frames.

    The provider call is not streamed: nexus waits for the complete answer and
    then sends a role chunk (carrying any text), one ``delta.tool_calls`` chunk
    per call with its ``index``, ``id``, ``type``, ``name``, and complete
    ``arguments``, a finish chunk, and ``data: [DONE]``. Clients that
    accumulate argument fragments by ``index`` read them unchanged.

    Args:
        response: Completed router response.

    Returns:
        SSE frames, each ending with a blank line.
    """
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time())

    def frame(delta: dict[str, object], finish_reason: str | None = None) -> str:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": response.model_used,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    frames = [frame({"role": "assistant", "content": response.content or None})]
    frames.extend(
        frame(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                ]
            }
        )
        for index, call in enumerate(response.tool_calls)
    )
    frames.append(frame({}, response.finish_reason))
    frames.append("data: [DONE]\n\n")
    return frames
