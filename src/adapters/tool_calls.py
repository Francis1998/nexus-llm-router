"""Provider-neutral helpers for translating OpenAI tool calls.

Adapters translate the OpenAI Chat Completions tool-calling shape (``tools``,
``tool_choice``, assistant ``tool_calls``, ``role: "tool"`` results) to their
provider's format and back. Nexus never executes tools: it only forwards the
model's calls to the client and the client's results to the model.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Collection, Sequence

from router.schemas import (
    ChatMessage,
    FinishReason,
    NamedToolChoice,
    ToolCall,
    ToolCallFunction,
    ToolConfig,
)

JsonObject = dict[str, object]

# LiteLLM's separator for Gemini thought signatures embedded in tool call ids, so
# ids minted by either gateway round-trip through the other.
THOUGHT_SIGNATURE_SEPARATOR = "__thought__"
# Gemini's documented placeholder for a replayed function call whose signature is
# unknown (for example one produced by another provider).
DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def arguments_object(arguments: str) -> JsonObject:
    """Parse JSON-encoded tool arguments into the object providers expect.

    The API rejects non-object arguments up front; anything else reaching an
    adapter (for example from a direct ``NexusRouter`` caller) is sent as ``{}``.

    Args:
        arguments: ``tool_calls[].function.arguments``.

    Returns:
        The decoded JSON object.
    """
    try:
        decoded = json.loads(arguments)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def provider_safe_id(tool_call_id: str) -> str:
    """Return a tool call id every provider accepts (``^[A-Za-z0-9_-]{1,64}$``).

    Ids from another provider (a long Gemini id carrying a thought signature,
    or one with ``.`` or ``:``) are replaced by a stable digest. The same id
    always maps to the same value, so a call and its result stay paired.

    Args:
        tool_call_id: Client-visible tool call id.

    Returns:
        The id itself when it is safe, otherwise ``call_<24 hex digits>``.
    """
    if _SAFE_ID.match(tool_call_id):
        return tool_call_id
    return "call_" + hashlib.sha256(tool_call_id.encode("utf-8")).hexdigest()[:24]


def stable_call_id(seed: str, index: int, name: str, arguments: str) -> str:
    """Return a deterministic id for a provider call that came without one.

    Args:
        seed: Digest of the request the call answers (differs every turn).
        index: Position of the call in the response.
        name: Function name.
        arguments: JSON-encoded arguments.

    Returns:
        ``call_<24 hex digits>``.
    """
    material = f"{seed}|{index}|{name}|{arguments}".encode()
    return "call_" + hashlib.sha256(material).hexdigest()[:24]


def request_seed(payload: object) -> str:
    """Return a digest of a provider payload, used to seed :func:`stable_call_id`."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def encode_thought_signature(tool_call_id: str, signature: object) -> str:
    """Embed a Gemini thought signature in a tool call id (``<id>__thought__<signature>``).

    Gemini 3 and later reject a replayed function call without the signature
    the model attached to it, and OpenAI clients only echo ``id``, ``type``,
    and ``function`` back. The signature is appended verbatim (standard base64,
    as Gemini returns it), which is LiteLLM's format, so a conversation can
    continue through either gateway.

    Args:
        tool_call_id: Base id of the call.
        signature: ``thoughtSignature`` of the Gemini part, if any.

    Returns:
        The id with the signature embedded, or ``tool_call_id`` without one.
    """
    if not isinstance(signature, str) or not signature:
        return tool_call_id
    return f"{tool_call_id}{THOUGHT_SIGNATURE_SEPARATOR}{signature}"


def decode_thought_signature(tool_call_id: str) -> str | None:
    """Return the Gemini thought signature embedded by :func:`encode_thought_signature`.

    Like LiteLLM, only a suffix that is strict standard base64 counts as a
    signature.

    Args:
        tool_call_id: Tool call id echoed by the client.

    Returns:
        The signature, or ``None`` when the id carries none.
    """
    _, separator, embedded = tool_call_id.partition(THOUGHT_SIGNATURE_SEPARATOR)
    if not separator or not embedded:
        return None
    try:
        base64.b64decode(embedded, validate=True)
    except ValueError:
        return None
    return embedded


def tool_finish_reason(
    stop_reason: object, length_values: Collection[str], tool_calls: Sequence[ToolCall]
) -> FinishReason:
    """Normalize a provider stop reason when the response may call tools.

    Args:
        stop_reason: Raw provider value.
        length_values: Provider values meaning the output-token cap was hit.
        tool_calls: Tool calls parsed from the response.

    Returns:
        ``length`` at the token cap (even with calls, whose arguments may be
        cut off), ``tool_calls`` when the model called tools, otherwise ``stop``.
    """
    if isinstance(stop_reason, str) and stop_reason in length_values:
        return "length"
    return "tool_calls" if tool_calls else "stop"


def openai_chat_messages(messages: Sequence[ChatMessage]) -> list[JsonObject]:
    """Serialize router messages for an OpenAI-compatible Chat Completions API.

    Text-only messages keep their ``{"role", "content"}`` shape; assistant tool
    calls and tool results add ``tool_calls`` / ``tool_call_id``.

    Args:
        messages: Router chat messages.

    Returns:
        Chat Completions ``messages``.
    """
    serialized: list[JsonObject] = []
    for message in messages:
        item: JsonObject = {"role": message.role, "content": message.content}
        if message.tool_calls:
            item["content"] = message.content or None
            item["tool_calls"] = [call.model_dump() for call in message.tool_calls]
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        serialized.append(item)
    return serialized


def openai_chat_tool_fields(tool_config: ToolConfig | None) -> JsonObject:
    """Return the Chat Completions ``tools`` / ``tool_choice`` / ``parallel_tool_calls`` fields.

    Args:
        tool_config: Validated tool settings, or ``None``.

    Returns:
        Fields to merge into the request payload (none for tool-free requests).
    """
    if tool_config is None:
        return {}
    fields: JsonObject = {
        "tools": [tool.model_dump(exclude_none=True) for tool in tool_config.tools]
    }
    choice = tool_config.tool_choice
    if isinstance(choice, NamedToolChoice):
        fields["tool_choice"] = choice.model_dump()
    elif choice is not None:
        fields["tool_choice"] = choice
    if tool_config.parallel_tool_calls is not None:
        fields["parallel_tool_calls"] = tool_config.parallel_tool_calls
    return fields


def parse_openai_tool_calls(message: object) -> list[ToolCall]:
    """Parse ``choices[0].message.tool_calls`` of an OpenAI-compatible response.

    Args:
        message: The ``choices[0].message`` object.

    Returns:
        Function calls with string arguments; malformed entries are skipped.
    """
    if not isinstance(message, dict):
        return []
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    calls: list[ToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict) or not isinstance(raw.get("function"), dict):
            continue
        function = raw["function"]
        call_id, name = raw.get("id"), function.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {})
        calls.append(
            ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments=arguments))
        )
    return calls


def history_tool_names(messages: Sequence[ChatMessage]) -> list[str]:
    """Return the distinct function names the conversation's assistant turns called."""
    names: list[str] = []
    for message in messages:
        for call in message.tool_calls or ():
            if call.function.name not in names:
                names.append(call.function.name)
    return names
