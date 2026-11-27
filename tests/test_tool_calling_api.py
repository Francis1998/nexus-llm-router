"""API tests for tool calling on POST /v1/chat/completions (offline, mock providers)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

import api.main as main
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.capability_catalog import ModelCapabilityCatalog
from router.config import RouterSettings
from router.engine import NexusRouter
from router.schemas import ChatMessage, ProviderResponse, ToolCall, ToolCallFunction, ToolConfig
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.tenant_rate_limiter import TenantRateLimiter

PROVIDERS = ("openai", "anthropic", "google", "moonshot")
WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "examples": ["Paris"]},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
}
TIME_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}},
}
QUESTION = {"role": "user", "content": "What's the weather in Paris?"}


class CountingAdapter(MockProviderAdapter):
    """Mock provider that counts calls; optionally answers every tool request with two calls."""

    def __init__(self, provider_name: str, *, parallel: bool = False, fail: bool = False) -> None:
        super().__init__(provider_name, fail=fail)
        self.calls = 0
        self._parallel = parallel

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        tool_config: ToolConfig | None = None,
    ) -> ProviderResponse:
        self.calls += 1
        if self._parallel and tool_config is not None:
            calls = [
                ToolCall(
                    id=f"call_{city}",
                    function=ToolCallFunction(
                        name="get_weather", arguments=json.dumps({"city": city})
                    ),
                )
                for city in ("Paris", "Tokyo")
            ]
            return ProviderResponse(
                content="Checking both cities.",
                model=model,
                input_tokens=10,
                output_tokens=8,
                cost_usd=0.0001,
                finish_reason="tool_calls",
                tool_calls=calls,
            )
        return await super().complete(model, messages, max_tokens, tool_config)


@dataclass
class Gateway:
    """Test client over counting mock providers."""

    client: TestClient
    adapters: dict[str, CountingAdapter]

    @property
    def provider_calls(self) -> int:
        return sum(adapter.calls for adapter in self.adapters.values())

    def chat(self, headers: dict[str, str] | None = None, **body: object) -> Response:
        payload = {"messages": [QUESTION], **body}
        return self.client.post("/v1/chat/completions", json=payload, headers=headers)


GatewayFactory = Callable[..., Gateway]


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GatewayFactory:
    """Build the app over counting mock providers with private state files."""

    def _build(
        *,
        capability_catalog: ModelCapabilityCatalog | None = None,
        cache: bool = False,
        parallel: bool = False,
        failing: bool = False,
    ) -> Gateway:
        settings = RouterSettings(
            _env_file=None,
            response_cache_enabled=cache,
            audit_log_path=str(tmp_path / "audit-log.jsonl"),
            spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
            idempotency_path=str(tmp_path / "idempotency.sqlite3"),
            feedback_path=str(tmp_path / "feedback.sqlite3"),
        )
        adapters = {
            name: CountingAdapter(name, parallel=parallel, fail=failing) for name in PROVIDERS
        }
        registry = AdapterRegistry(dict(adapters))
        router = NexusRouter(settings, registry, capability_catalog=capability_catalog)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=1000, refill_per_second=1.0),
        )
        response_cache = ResponseCache(enabled=cache)
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: response_cache)
        return Gateway(TestClient(main.app, raise_server_exceptions=False), adapters)

    return _build


def _sse(body: str) -> list[Any]:
    frames: list[Any] = []
    for block in body.split("\n\n"):
        line = block.strip()
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            frames.append(payload if payload == "[DONE]" else json.loads(payload))
    return frames


def test_tool_round_trip_returns_tool_calls_then_a_final_answer(gateway: GatewayFactory) -> None:
    """tools -> tool_calls (content null) -> tool result -> final answer."""
    gw = gateway()
    first = gw.chat(tools=[WEATHER_TOOL])
    assert first.status_code == 200, first.text
    choice = first.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] is None
    [call] = message["tool_calls"]
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"] == {
        "name": "get_weather",
        "arguments": json.dumps({"city": "Paris", "unit": "celsius"}),
    }
    assert "tool_calls" not in first.json()["router"]

    result = {"role": "tool", "tool_call_id": call["id"], "content": '{"temp_c": 18}'}
    second = gw.chat(
        messages=[QUESTION, {**message, "content": None}, result], tools=[WEATHER_TOOL]
    )
    assert second.status_code == 200, second.text
    final = second.json()["choices"][0]
    assert final["finish_reason"] == "stop"
    assert final["message"] == {
        "role": "assistant",
        "content": final["message"]["content"],
    }
    assert final["message"]["content"].endswith('get_weather returned {"temp_c": 18}')


def test_tool_choice_forces_a_function_or_disables_tools(gateway: GatewayFactory) -> None:
    """A named tool_choice picks the function; ``none`` answers with text."""
    gw = gateway()
    forced = gw.chat(
        tools=[WEATHER_TOOL, TIME_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_time"}},
        parallel_tool_calls=False,
    ).json()
    assert forced["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_time"
    plain = gw.chat(tools=[WEATHER_TOOL], tool_choice="none").json()["choices"][0]
    assert (plain["finish_reason"], "tool_calls" in plain["message"]) == ("stop", False)


def test_text_only_responses_keep_their_shape(gateway: GatewayFactory) -> None:
    """Requests without tools get no ``tool_calls`` key, as before."""
    body = gateway().chat(tools=[]).json()
    assert set(body["choices"][0]["message"]) == {"role", "content"}
    assert body["choices"][0]["finish_reason"] == "stop"


def _named_tool(name: str) -> dict[str, Any]:
    return {**WEATHER_TOOL, "function": {**WEATHER_TOOL["function"], "name": name}}


ASSISTANT_CALL = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ],
}
RESULT = {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 18}'}


def _with_call(arguments: str = '{"city": "Paris"}', **fields: object) -> dict[str, Any]:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": arguments},
    }
    return {**ASSISTANT_CALL, "tool_calls": [{**call, **fields}]}


@pytest.mark.parametrize(
    ("body", "param", "code"),
    [
        ({"tools": [_named_tool("get weather")]}, "tools[0].function.name", "invalid_value"),
        ({"tools": [_named_tool("x" * 65)]}, "tools[0].function.name", "invalid_value"),
        (
            {
                "tools": [
                    {**WEATHER_TOOL, "function": {"name": "f", "parameters": {"type": "string"}}}
                ]
            },
            "tools[0].function.parameters",
            "invalid_value",
        ),
        ({"tools": [{**WEATHER_TOOL, "type": "web_search"}]}, "tools[0].type", "invalid_value"),
        ({"tools": [_named_tool(f"f{i}") for i in range(129)]}, "tools", "invalid_value"),
        ({"tools": [WEATHER_TOOL, WEATHER_TOOL]}, "tools", "invalid_value"),
        ({"tools": [WEATHER_TOOL], "tool_choice": "any"}, "tool_choice", "invalid_value"),
        (
            {
                "tools": [WEATHER_TOOL],
                "tool_choice": {"type": "function", "function": {"name": "nope"}},
            },
            "tool_choice",
            "invalid_value",
        ),
        ({"tool_choice": "required"}, "tool_choice", "invalid_value"),
        (
            {"tools": [WEATHER_TOOL], "parallel_tool_calls": "yes"},
            "parallel_tool_calls",
            "invalid_value",
        ),
        ({"functions": [WEATHER_TOOL["function"]]}, "functions", "unsupported_parameter"),
        ({"function_call": "auto"}, "function_call", "unsupported_parameter"),
        ({"messages": [QUESTION, RESULT]}, "messages[1].tool_call_id", "invalid_value"),
        (
            {"messages": [QUESTION, ASSISTANT_CALL, {"role": "tool", "content": "18C"}]},
            "messages[2].tool_call_id",
            "missing_required_parameter",
        ),
        (
            {"messages": [QUESTION, ASSISTANT_CALL, {**RESULT, "tool_call_id": "call_9"}]},
            "messages[2].tool_call_id",
            "invalid_value",
        ),
        (
            {"messages": [QUESTION, ASSISTANT_CALL, RESULT, RESULT]},
            "messages[3].tool_call_id",
            "invalid_value",
        ),
        ({"messages": [QUESTION, ASSISTANT_CALL]}, "messages[1].tool_calls", "invalid_value"),
        (
            {"messages": [QUESTION, ASSISTANT_CALL, QUESTION]},
            "messages[1].tool_calls",
            "invalid_value",
        ),
        (
            {"messages": [QUESTION, ASSISTANT_CALL, RESULT, QUESTION, RESULT]},
            "messages[4].tool_call_id",
            "invalid_value",
        ),
        (
            {"messages": [QUESTION, _with_call("[1, 2]"), RESULT]},
            "messages[1].tool_calls[0].function.arguments",
            "invalid_value",
        ),
        (
            {"messages": [QUESTION, _with_call("{not json"), RESULT]},
            "messages[1].tool_calls[0].function.arguments",
            "invalid_value",
        ),
        (
            {"messages": [QUESTION, _with_call(type="custom"), RESULT]},
            "messages[1].tool_calls[0].type",
            "invalid_value",
        ),
        (
            {"messages": [{**QUESTION, "tool_calls": ASSISTANT_CALL["tool_calls"]}, RESULT]},
            "messages[0].tool_calls",
            "invalid_value",
        ),
        (
            {"messages": [{**QUESTION, "tool_call_id": "call_1"}]},
            "messages[0].tool_call_id",
            "invalid_value",
        ),
    ],
)
def test_invalid_tool_requests_are_openai_style_400s(
    gateway: GatewayFactory, body: dict[str, Any], param: str, code: str
) -> None:
    """Every tool-calling request error is a 400 naming the parameter; nothing is dispatched."""
    gw = gateway()
    response = gw.chat(**body)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert (error["type"], error["param"], error["code"]) == ("invalid_request_error", param, code)
    assert '"city": "Paris"' not in error["message"]
    assert gw.provider_calls == 0


def test_other_validation_errors_keep_fastapi_422s(gateway: GatewayFactory) -> None:
    """Errors outside the tool fields keep the route's existing 422 responses."""
    response = gateway().chat(max_tokens=0, tools=[WEATHER_TOOL])
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize("stream", [False, True])
def test_requests_no_routed_model_can_serve_get_a_400_and_free_their_idempotency_key(
    gateway: GatewayFactory, stream: bool
) -> None:
    """``tool_calling_unsupported`` is an OpenAI-style 400; retries are not blocked by a 409."""
    gw = gateway(capability_catalog=ModelCapabilityCatalog(auto_seed=False))
    headers = {"Idempotency-Key": "tool-retry-1"}
    for _ in range(2):
        response = gw.chat(headers=headers, tools=[WEATHER_TOOL], stream=stream)
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert (error["type"], error["code"]) == (
            "invalid_request_error",
            "tool_calling_unsupported",
        )
        assert "supports tool calling" in error["message"]
    assert gw.provider_calls == 0
    text = gw.chat(stream=stream)
    assert text.status_code == 200


def test_streamed_tool_calls_follow_the_openai_chunk_shape(gateway: GatewayFactory) -> None:
    """Role chunk, one ``delta.tool_calls`` chunk per call, finish chunk, ``[DONE]``."""
    gw = gateway(parallel=True)
    response = gw.chat(tools=[WEATHER_TOOL], stream=True)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _sse(response.text)
    assert frames[-1] == "[DONE]"
    chunks = frames[:-1]
    assert {chunk["object"] for chunk in chunks} == {"chat.completion.chunk"}
    assert len({chunk["id"] for chunk in chunks}) == 1
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    assert deltas[0] == {"role": "assistant", "content": "Checking both cities."}
    assert deltas[1:3] == [
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": f"call_{city}",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": json.dumps({"city": city})},
                }
            ]
        }
        for index, city in enumerate(("Paris", "Tokyo"))
    ]
    assert deltas[3] == {}
    assert [chunk["choices"][0]["finish_reason"] for chunk in chunks] == [
        None,
        None,
        None,
        "tool_calls",
    ]
    assert gw.provider_calls == 1


def test_streamed_tool_only_answer_has_null_content_and_streamed_final_answer(
    gateway: GatewayFactory,
) -> None:
    """A tool-only answer streams ``content: null``; the follow-up answer streams as text."""
    gw = gateway()
    first = _sse(gw.chat(tools=[WEATHER_TOOL], stream=True).text)
    assert first[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    call = first[1]["choices"][0]["delta"]["tool_calls"][0]
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{key: call[key] for key in ("id", "type", "function")}],
    }
    result = {"role": "tool", "tool_call_id": call["id"], "content": "18C and sunny"}
    final = _sse(
        gw.chat(messages=[QUESTION, assistant, result], tools=[WEATHER_TOOL], stream=True).text
    )
    assert final[0]["choices"][0]["delta"]["content"].endswith("get_weather returned 18C and sunny")
    assert final[-2]["choices"][0]["finish_reason"] == "stop"


def test_streamed_tool_requests_report_routing_failures_as_http_errors(
    gateway: GatewayFactory,
) -> None:
    """Nothing was streamed yet, so a failed provider call is a plain HTTP error."""
    response = gateway(failing=True).chat(tools=[WEATHER_TOOL], stream=True)
    assert response.status_code == 503
    assert "all provider attempts failed" in response.json()["detail"]


def test_response_cache_separates_tool_sets_and_tool_turns(gateway: GatewayFactory) -> None:
    """Identical tool requests hit the cache; other tools or tool_choice values do not."""
    gw = gateway(cache=True)
    first = gw.chat(tools=[WEATHER_TOOL]).json()
    assert gw.chat(tools=[WEATHER_TOOL]).json()["choices"] == first["choices"]
    assert gw.provider_calls == 1
    gw.chat(tools=[WEATHER_TOOL, TIME_TOOL])
    gw.chat(tools=[WEATHER_TOOL], tool_choice="required")
    gw.chat(tools=[WEATHER_TOOL], parallel_tool_calls=False)
    gw.chat()
    assert gw.provider_calls == 5


def test_idempotent_retries_replay_the_same_tool_calls(gateway: GatewayFactory) -> None:
    """A retried ``Idempotency-Key`` returns the stored tool calls without a new dispatch."""
    gw = gateway()
    headers = {"Idempotency-Key": "tool-replay-1"}
    first = gw.chat(headers=headers, tools=[WEATHER_TOOL])
    replay = gw.chat(headers=headers, tools=[WEATHER_TOOL])
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert replay.json()["choices"][0]["message"]["tool_calls"]
    assert gw.provider_calls == 1
