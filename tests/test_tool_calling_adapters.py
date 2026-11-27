"""Offline tests for tool-call translation in every provider adapter."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.anthropic import AnthropicAdapter
from adapters.base import ProviderError
from adapters.google import GoogleGeminiAdapter
from adapters.mock import MockProviderAdapter, schema_example_arguments
from adapters.moonshot import MoonshotAdapter
from adapters.openai import OpenAIAdapter
from adapters.tool_calls import (
    DUMMY_THOUGHT_SIGNATURE,
    decode_thought_signature,
    encode_thought_signature,
    provider_safe_id,
)
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_EVERYDAY_MODEL,
)
from router.schemas import (
    ChatMessage,
    NamedToolChoice,
    ToolCall,
    ToolCallFunction,
    ToolChoice,
    ToolChoiceFunction,
    ToolConfig,
    ToolDefinition,
)

WEATHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "examples": ["Paris"]},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["city"],
}


def _tool(
    name: str, description: str | None = None, parameters: dict[str, Any] | None = None
) -> ToolDefinition:
    return ToolDefinition.model_validate(
        {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        }
    )


def _config(
    *names: str, tool_choice: ToolChoice | None = None, parallel_tool_calls: bool | None = None
) -> ToolConfig:
    tools = tuple(_tool(name, f"{name} tool", WEATHER_SCHEMA) for name in names or ("get_weather",))
    return ToolConfig(tools=tools, tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls)


def _named(name: str) -> NamedToolChoice:
    return NamedToolChoice(function=ToolChoiceFunction(name=name))


def _call(
    call_id: str, name: str = "get_weather", arguments: str = '{"city": "Paris"}'
) -> ToolCall:
    return ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments=arguments))


def _round_trip(*call_ids: str) -> list[ChatMessage]:
    """User question, assistant tool calls, and one tool result per call."""
    return [
        ChatMessage(role="system", content="Be brief."),
        ChatMessage(role="user", content="Weather in Paris and Tokyo?"),
        ChatMessage(role="assistant", content="", tool_calls=[_call(cid) for cid in call_ids]),
        *[
            ChatMessage(role="tool", tool_call_id=cid, content=f'{{"temp_c": {index}}}')
            for index, cid in enumerate(call_ids)
        ],
    ]


@contextmanager
def _provider(module: str, body: dict[str, Any]) -> Iterator[list[dict[str, Any]]]:
    """Patch ``<module>.httpx.AsyncClient``; yield the captured ``{url, json}`` posts."""
    posts: list[dict[str, Any]] = []

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        posts.append({"url": url, "json": kwargs["json"]})
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = body
        return response

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = fake_post
    with patch(f"adapters.{module}.httpx.AsyncClient", return_value=client):
        yield posts


_CHAT_BODY = {
    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}
_RESPONSES_CALL_BODY = {
    "output": [
        {"type": "reasoning", "summary": []},
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Checking."}],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        },
    ],
    "usage": {"input_tokens": 40, "output_tokens": 12},
}


async def test_openai_tool_free_request_stays_on_chat_completions() -> None:
    """Requests without tools keep the exact Chat Completions payload they had before."""
    adapter = OpenAIAdapter(api_key="key", timeout_seconds=5.0)
    with _provider("openai", _CHAT_BODY) as posts:
        response = await adapter.complete(
            OPENAI_EVERYDAY_MODEL, [ChatMessage(role="user", content="hello")], 32
        )
    assert posts[0]["url"] == "https://api.openai.com/v1/chat/completions"
    assert posts[0]["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert "tools" not in posts[0]["json"]
    assert (response.content, response.tool_calls, response.finish_reason) == ("hi", [], "stop")


async def test_openai_tool_request_goes_through_the_responses_api() -> None:
    """GPT-6.2 needs the Responses API for tools: tools, choices, and calls are translated."""
    adapter = OpenAIAdapter(api_key="key", timeout_seconds=5.0)
    config = _config(
        "get_weather", "get_time", tool_choice=_named("get_weather"), parallel_tool_calls=False
    )
    with _provider("openai", _RESPONSES_CALL_BODY) as posts:
        response = await adapter.complete(
            OPENAI_EVERYDAY_MODEL,
            [ChatMessage(role="system", content="Be brief."), ChatMessage(content="Paris?")],
            64,
            tool_config=config,
        )
    payload = posts[0]["json"]
    assert posts[0]["url"] == "https://api.openai.com/v1/responses"
    assert payload["store"] is False
    assert payload["max_output_tokens"] == 64
    assert payload["tools"][0] == {
        "type": "function",
        "name": "get_weather",
        "description": "get_weather tool",
        "parameters": WEATHER_SCHEMA,
        "strict": False,
    }
    assert payload["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert payload["parallel_tool_calls"] is False
    assert payload["input"] == [
        {"type": "message", "role": "developer", "content": "Be brief."},
        {"type": "message", "role": "user", "content": "Paris?"},
    ]
    assert response.content == "Checking."
    assert response.tool_calls == [_call("call_abc", arguments='{"city":"Paris"}')]
    assert (response.finish_reason, response.input_tokens, response.output_tokens) == (
        "tool_calls",
        40,
        12,
    )


async def test_openai_replays_calls_and_results_as_responses_items() -> None:
    """Assistant tool calls and tool results become paired function_call(_output) items."""
    adapter = OpenAIAdapter(api_key="key", timeout_seconds=5.0)
    foreign_id = "call_gemini__thought__" + "A" * 80  # longer than the 64-character limit
    body = {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "Mild."}]}],
        "usage": {"input_tokens": 50, "output_tokens": 4},
    }
    with _provider("openai", body) as posts:
        response = await adapter.complete(
            OPENAI_EVERYDAY_MODEL, _round_trip("call_1", foreign_id), 64, tool_config=_config()
        )
    items = posts[0]["json"]["input"]
    safe_id = provider_safe_id(foreign_id)
    assert safe_id != foreign_id and len(safe_id) <= 64
    assert items[2:] == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
        },
        {
            "type": "function_call",
            "call_id": safe_id,
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": '{"temp_c": 0}'},
        {"type": "function_call_output", "call_id": safe_id, "output": '{"temp_c": 1}'},
    ]
    assert (response.content, response.finish_reason) == ("Mild.", "stop")


async def test_openai_responses_output_cap_reports_length() -> None:
    """``incomplete_details.reason == max_output_tokens`` maps to ``length``."""
    adapter = OpenAIAdapter(api_key="key", timeout_seconds=5.0)
    body = {**_RESPONSES_CALL_BODY, "incomplete_details": {"reason": "max_output_tokens"}}
    with _provider("openai", body):
        response = await adapter.complete(
            OPENAI_EVERYDAY_MODEL, [ChatMessage(content="x")], 64, tool_config=_config()
        )
    assert response.finish_reason == "length"


_ANTHROPIC_TOOL_USE_BODY = {
    "content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 30, "output_tokens": 9},
}


@pytest.mark.parametrize(
    ("tool_choice", "parallel", "expected"),
    [
        (None, None, None),
        ("auto", None, {"type": "auto"}),
        ("required", None, {"type": "any"}),
        ("none", None, {"type": "none"}),
        (_named("get_weather"), None, {"type": "tool", "name": "get_weather"}),
        (None, False, {"type": "auto", "disable_parallel_tool_use": True}),
        ("required", False, {"type": "any", "disable_parallel_tool_use": True}),
        ("none", False, {"type": "none"}),
    ],
)
async def test_anthropic_translates_tools_and_tool_choice(
    tool_choice: ToolChoice | None, parallel: bool | None, expected: dict[str, Any] | None
) -> None:
    """Tools become ``input_schema`` definitions; ``tool_choice`` maps to auto/any/tool/none."""
    adapter = AnthropicAdapter(api_key="key", timeout_seconds=5.0)
    config = _config(tool_choice=tool_choice, parallel_tool_calls=parallel)
    with _provider("anthropic", _ANTHROPIC_TOOL_USE_BODY) as posts:
        response = await adapter.complete(
            ANTHROPIC_SAFETY_MODEL, [ChatMessage(content="Paris?")], 64, tool_config=config
        )
    payload = posts[0]["json"]
    assert payload["tools"] == [
        {"name": "get_weather", "description": "get_weather tool", "input_schema": WEATHER_SCHEMA}
    ]
    assert payload.get("tool_choice") == expected
    assert response.content == "Let me check."
    assert response.tool_calls == [_call("toolu_01", arguments='{"city": "Paris"}')]
    assert response.finish_reason == "tool_calls"


async def test_anthropic_replays_tool_use_and_merges_consecutive_tool_results() -> None:
    """Calls become ``tool_use`` blocks; consecutive results share one user turn."""
    adapter = AnthropicAdapter(api_key="key", timeout_seconds=5.0)
    body = {"content": [{"type": "text", "text": "Mild."}], "stop_reason": "end_turn"}
    with _provider("anthropic", body) as posts:
        response = await adapter.complete(
            ANTHROPIC_SAFETY_MODEL, _round_trip("call_1", "call.2"), 64, tool_config=_config()
        )
    payload = posts[0]["json"]
    assert payload["system"] == "Be brief."
    safe_second = provider_safe_id("call.2")
    assert payload["messages"][1:] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
                {
                    "type": "tool_use",
                    "id": safe_second,
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": '{"temp_c": 0}'},
                {"type": "tool_result", "tool_use_id": safe_second, "content": '{"temp_c": 1}'},
            ],
        },
    ]
    assert (response.content, response.tool_calls, response.finish_reason) == ("Mild.", [], "stop")


async def test_anthropic_history_without_tools_declares_called_functions_with_choice_none() -> None:
    """Anthropic rejects tool blocks without ``tools``, so replayed calls get minimal ones."""
    adapter = AnthropicAdapter(api_key="key", timeout_seconds=5.0)
    body = {"content": [{"type": "text", "text": "Mild."}], "stop_reason": "end_turn"}
    with _provider("anthropic", body) as posts:
        await adapter.complete(ANTHROPIC_SAFETY_MODEL, _round_trip("call_1"), 64)
    payload = posts[0]["json"]
    assert payload["tools"] == [
        {"name": "get_weather", "input_schema": {"type": "object", "properties": {}}}
    ]
    assert payload["tool_choice"] == {"type": "none"}


async def test_anthropic_max_tokens_with_partial_tool_use_reports_length() -> None:
    """A tool call cut off at the token cap is reported as ``length``, not ``tool_calls``."""
    adapter = AnthropicAdapter(api_key="key", timeout_seconds=5.0)
    body = {**_ANTHROPIC_TOOL_USE_BODY, "stop_reason": "max_tokens"}
    with _provider("anthropic", body):
        response = await adapter.complete(
            ANTHROPIC_SAFETY_MODEL, [ChatMessage(content="Paris?")], 8, tool_config=_config()
        )
    assert response.finish_reason == "length"


SIGNATURE = base64.b64encode(bytes(range(250, 256)) + b"sig/+").decode()  # has '+' and '/'


def _gemini_body(*parts: dict[str, Any], finish: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": list(parts)}, "finishReason": finish}
        ],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5},
    }


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        (None, None),
        ("auto", {"mode": "AUTO"}),
        ("required", {"mode": "ANY"}),
        ("none", {"mode": "NONE"}),
        (_named("get_time"), {"mode": "ANY", "allowedFunctionNames": ["get_time"]}),
    ],
)
async def test_gemini_translates_function_declarations_and_calling_config(
    tool_choice: ToolChoice | None, expected: dict[str, Any] | None
) -> None:
    """Tools become functionDeclarations; tool_choice maps to functionCallingConfig."""
    adapter = GoogleGeminiAdapter(api_key="key", timeout_seconds=5.0)
    config = _config("get_weather", "get_time", tool_choice=tool_choice)
    with _provider("google", _gemini_body({"text": "ok"})) as posts:
        await adapter.complete(GEMINI_FLASH_MODEL, [ChatMessage(content="hi")], 64, config)
    payload = posts[0]["json"]
    assert payload["tools"][0]["functionDeclarations"][0] == {
        "name": "get_weather",
        "description": "get_weather tool",
        "parametersJsonSchema": WEATHER_SCHEMA,
    }
    calling = payload.get("toolConfig", {}).get("functionCallingConfig")
    assert calling == expected


async def test_gemini_function_calls_get_stable_ids_and_carry_thought_signatures() -> None:
    """Calls without ids get ids stable for one request; the signature rides in the id."""
    adapter = GoogleGeminiAdapter(api_key="key", timeout_seconds=5.0)
    body = _gemini_body(
        {
            "functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
            "thoughtSignature": SIGNATURE,
        },
        {"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}},
    )
    question = [ChatMessage(content="Paris and Tokyo?")]
    with _provider("google", body):
        first = await adapter.complete(GEMINI_FLASH_MODEL, question, 64, _config())
        again = await adapter.complete(GEMINI_FLASH_MODEL, question, 64, _config())
        other = await adapter.complete(
            GEMINI_FLASH_MODEL, [ChatMessage(content="Rome?")], 64, _config()
        )
    assert first.finish_reason == "tool_calls"
    assert [call.function.arguments for call in first.tool_calls] == [
        '{"city": "Paris"}',
        '{"city": "Tokyo"}',
    ]
    assert first.tool_calls == again.tool_calls
    assert {call.id for call in first.tool_calls}.isdisjoint(call.id for call in other.tool_calls)
    assert decode_thought_signature(first.tool_calls[0].id) == SIGNATURE
    assert decode_thought_signature(first.tool_calls[1].id) is None
    assert all(
        set(call.id) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        for call in first.tool_calls
    )


async def test_gemini_replays_signatures_and_names_function_responses() -> None:
    """The next turn returns the signature and answers each call with a named functionResponse."""
    adapter = GoogleGeminiAdapter(api_key="key", timeout_seconds=5.0)
    signed = encode_thought_signature("call_a", SIGNATURE)
    messages = [
        ChatMessage(content="Paris and Tokyo?"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[_call(signed), _call("call_b", "get_time", "{}")],
        ),
        ChatMessage(role="tool", tool_call_id=signed, content='{"temp_c": 18}'),
        ChatMessage(role="tool", tool_call_id="call_b", content="09:00"),
    ]
    with _provider("google", _gemini_body({"text": "Mild."})) as posts:
        response = await adapter.complete(GEMINI_FLASH_MODEL, messages, 64, _config())
    contents = posts[0]["json"]["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [
            {
                "functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
                "thoughtSignature": SIGNATURE,
            },
            {"functionCall": {"name": "get_time", "args": {}}},
        ],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [
            {"functionResponse": {"name": "get_weather", "response": {"temp_c": 18}}},
            {"functionResponse": {"name": "get_time", "response": {"output": "09:00"}}},
        ],
    }
    assert (response.content, response.finish_reason) == ("Mild.", "stop")


async def test_gemini_unsigned_replayed_calls_get_the_documented_placeholder() -> None:
    """Calls from another provider carry no signature: the first gets Gemini's placeholder."""
    adapter = GoogleGeminiAdapter(api_key="key", timeout_seconds=5.0)
    with _provider("google", _gemini_body({"text": "Mild."})) as posts:
        await adapter.complete(GEMINI_FLASH_MODEL, _round_trip("toolu_1", "toolu_2"), 64)
    parts = posts[0]["json"]["contents"][1]["parts"]
    assert parts[0]["thoughtSignature"] == DUMMY_THOUGHT_SIGNATURE
    assert "thoughtSignature" not in parts[1]


async def test_gemini_parallel_tool_calls_false_keeps_only_the_first_call() -> None:
    """Gemini has no switch for parallel calls, so nexus keeps the first one."""
    adapter = GoogleGeminiAdapter(api_key="key", timeout_seconds=5.0)
    body = _gemini_body(
        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}},
        {"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}},
    )
    with _provider("google", body):
        response = await adapter.complete(
            GEMINI_FLASH_MODEL, [ChatMessage(content="x")], 64, _config(parallel_tool_calls=False)
        )
    assert [call.function.arguments for call in response.tool_calls] == ['{"city": "Paris"}']


async def test_moonshot_passes_tool_fields_through_unchanged() -> None:
    """Kimi's API is OpenAI-compatible: tool fields and messages pass through."""
    adapter = MoonshotAdapter("key", "https://api.moonshot.ai/v1", 5.0)
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "get_weather:0",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }
    config = _config(tool_choice="required", parallel_tool_calls=True)
    with _provider("moonshot", body) as posts:
        response = await adapter.complete(
            MOONSHOT_BALANCED_MODEL, _round_trip("call_1"), 64, tool_config=config
        )
    payload = posts[0]["json"]
    assert payload["tools"] == [tool.model_dump(exclude_none=True) for tool in config.tools]
    assert (payload["tool_choice"], payload["parallel_tool_calls"]) == ("required", True)
    assert payload["messages"][2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [_call("call_1").model_dump()],
    }
    assert payload["messages"][3] == {
        "role": "tool",
        "content": '{"temp_c": 0}',
        "tool_call_id": "call_1",
    }
    assert response.tool_calls == [_call("get_weather:0", arguments='{"city":"Paris"}')]
    assert (response.content, response.finish_reason) == ("", "tool_calls")


async def test_mock_emulates_a_deterministic_tool_call_from_the_schema() -> None:
    """The mock calls the first tool with schema-derived arguments and stable ids."""
    adapter = MockProviderAdapter("openai")
    question = [ChatMessage(content="Weather in Paris?")]
    first = await adapter.complete(OPENAI_EVERYDAY_MODEL, question, 64, _config())
    again = await adapter.complete(OPENAI_EVERYDAY_MODEL, question, 64, _config())
    assert (first.content, first.finish_reason) == ("", "tool_calls")
    assert [call.function.name for call in first.tool_calls] == ["get_weather"]
    assert json.loads(first.tool_calls[0].function.arguments) == {"city": "Paris"}
    assert first.tool_calls == again.tool_calls
    assert first.tool_calls[0].id.startswith("call_")


async def test_mock_answers_from_tool_results_unless_a_call_is_required() -> None:
    """``auto`` answers with text after tool results; ``required``/named keep calling."""
    adapter = MockProviderAdapter("openai")
    conversation = _round_trip("call_1")
    answer = await adapter.complete(OPENAI_EVERYDAY_MODEL, conversation, 64, _config())
    assert answer.tool_calls == []
    assert answer.finish_reason == "stop"
    assert answer.content == (
        f'[{OPENAI_EVERYDAY_MODEL}] answer from tool results: get_weather returned {{"temp_c": 0}}'
    )
    for choice in ("required", _named("get_time")):
        config = _config("get_weather", "get_time", tool_choice=choice)
        forced = await adapter.complete(OPENAI_EVERYDAY_MODEL, conversation, 64, config)
        expected = "get_time" if choice != "required" else "get_weather"
        assert [call.function.name for call in forced.tool_calls] == [expected]
    plain = await adapter.complete(
        OPENAI_EVERYDAY_MODEL, [ChatMessage(content="hi")], 64, _config(tool_choice="none")
    )
    assert (plain.content, plain.tool_calls) == (f"[{OPENAI_EVERYDAY_MODEL}] routed response", [])


async def test_mock_stream_carries_tool_calls_in_the_final_response() -> None:
    """Streams yield text chunks, then one ProviderResponse holding the tool calls."""
    adapter = MockProviderAdapter("openai")
    items = [
        item
        async for item in adapter.stream(
            OPENAI_EVERYDAY_MODEL, [ChatMessage(content="Paris?")], 64, _config()
        )
    ]
    final = items[-1]
    assert not isinstance(final, str)
    assert [call.function.name for call in final.tool_calls] == ["get_weather"]
    assert all(isinstance(item, str) and not item for item in items[:-1])


def test_schema_example_arguments_follow_schema_hints() -> None:
    """const/enum/default/examples win over type placeholders; required limits the keys."""
    schema = {
        "type": "object",
        "properties": {
            "mode": {"const": "fast"},
            "unit": {"enum": ["c", "f"]},
            "limit": {"type": "integer", "default": 5},
            "city": {"type": "string", "examples": ["Paris", "Rome"]},
            "count": {"type": ["integer", "null"], "minimum": 2},
            "flag": {"type": "boolean"},
            "tags": {"type": "array"},
            "where": {"type": "object", "properties": {"zip": {"type": "string"}}},
            "note": {"type": "string"},
            "skipped": {"type": "string"},
        },
        "required": ["mode", "unit", "limit", "city", "count", "flag", "tags", "where", "note"],
    }
    assert schema_example_arguments(schema) == {
        "mode": "fast",
        "unit": "c",
        "limit": 5,
        "city": "Paris",
        "count": 2,
        "flag": True,
        "tags": [],
        "where": {"zip": "mock-zip"},
        "note": "mock-note",
    }
    assert schema_example_arguments({"type": "object"}) == {}


def test_provider_safe_ids_are_stable_and_pass_safe_ids_through() -> None:
    """Unsafe ids map to one stable ``call_<hex>`` digest; safe ids are untouched."""
    assert provider_safe_id("toolu_01A-b") == "toolu_01A-b"
    unsafe = provider_safe_id("get_weather:0")
    assert unsafe == provider_safe_id("get_weather:0")
    assert unsafe.startswith("call_") and len(unsafe) == 29


def test_thought_signatures_round_trip_through_url_safe_ids() -> None:
    """Standard-base64 signatures survive the id encoding byte for byte."""
    encoded = encode_thought_signature("call_x", SIGNATURE)
    assert "+" not in encoded and "/" not in encoded and "=" not in encoded
    assert decode_thought_signature(encoded) == SIGNATURE
    assert encode_thought_signature("call_x", None) == "call_x"
    assert decode_thought_signature("call_x") is None


async def test_gemini_sends_out_of_order_results_in_the_calls_order() -> None:
    """Name-only functionResponses pair by position, so they follow the calls' order."""
    adapter = GoogleGeminiAdapter(api_key="key", timeout_seconds=5.0)
    messages = [
        ChatMessage(content="Paris and Tokyo?"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                _call("call_paris", arguments='{"city": "Paris"}'),
                _call("call_tokyo", arguments='{"city": "Tokyo"}'),
                _call("call_time", "get_time", "{}"),
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call_time", content='{"city": "UTC"}'),
        ChatMessage(role="tool", tool_call_id="call_tokyo", content='{"city": "Tokyo"}'),
        ChatMessage(role="tool", tool_call_id="call_paris", content='{"city": "Paris"}'),
    ]
    with _provider("google", _gemini_body({"text": "Mild."})) as posts:
        await adapter.complete(GEMINI_FLASH_MODEL, messages, 64, _config())
    parts = posts[0]["json"]["contents"][2]["parts"]
    assert [part["functionResponse"]["response"]["city"] for part in parts] == [
        "Paris",
        "Tokyo",
        "UTC",
    ]


async def test_openai_tool_requests_below_the_responses_minimum_are_rejected() -> None:
    """The Responses API needs 16 output tokens; nexus never silently raises the client's cap."""
    adapter = OpenAIAdapter(api_key="key", timeout_seconds=5.0)
    with _provider("openai", _RESPONSES_CALL_BODY) as posts:
        with pytest.raises(ProviderError) as raised:
            await adapter.complete(
                OPENAI_EVERYDAY_MODEL, [ChatMessage(content="x")], 15, tool_config=_config()
            )
        await adapter.complete(
            OPENAI_EVERYDAY_MODEL, [ChatMessage(content="x")], 16, tool_config=_config()
        )
    assert raised.value.status_code == 400
    assert [post["json"]["max_output_tokens"] for post in posts] == [16]


def test_thought_signatures_use_the_litellm_id_format() -> None:
    """``<id>__thought__<standard base64>`` verbatim, so ids round-trip through LiteLLM."""
    encoded = encode_thought_signature("call_x", SIGNATURE)
    assert encoded == f"call_x__thought__{SIGNATURE}"
    assert decode_thought_signature(encoded) == SIGNATURE
    assert decode_thought_signature(f"call_litellm__thought__{SIGNATURE}") == SIGNATURE
    assert decode_thought_signature("call_x__thought__not base64!") is None
