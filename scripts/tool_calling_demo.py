"""Tool-calling round trip against a running nexus: tools -> tool_calls -> tool result -> answer.

Start nexus with mock providers, then run the demo::

    PYTHONPATH=src python scripts/serve_mock.py --port 8765 &
    PYTHONPATH=src python scripts/tool_calling_demo.py --base-url http://127.0.0.1:8765

The client offers a ``get_weather`` function, receives the model's tool call
(nexus never runs tools), runs its own local ``get_weather``, sends the result
back, and gets the final answer. It then streams the first turn again to show
the ``delta.tool_calls`` chunks, and sends a tool result that answers no call to
show the 400. ``--gif`` renders the printed transcript with the shared demo-GIF
helpers (``create_demo_gif`` / ``feedback_loop_demo``).
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from create_demo_gif import ACCENT, MUTED, SUCCESS, TEXT, WARN, Line
from feedback_loop_demo import render_gif

HEADERS = {"Authorization": "Bearer sk-demo-local"}
QUESTION = {"role": "user", "content": "What's the weather in Paris right now?"}
WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name", "examples": ["Paris"]},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
}


class DemoError(RuntimeError):
    """Raised when the server does not behave like nexus with mock providers."""


def get_weather(city: str, unit: str) -> dict[str, object]:
    """The client's own tool: a canned local weather lookup (nexus never runs tools)."""
    return {"city": city, "temp": 18 if unit == "celsius" else 64, "unit": unit, "sky": "clear"}


class Session:
    """HTTP client for the demo plus the transcript it prints."""

    def __init__(self, base_url: str) -> None:
        """Open the client.

        Args:
            base_url: Server root, for example ``http://127.0.0.1:8765``.
        """
        self._client = httpx.Client(base_url=base_url, timeout=30.0)
        self.steps: list[list[Line]] = []

    def step(self, title: str) -> None:
        """Start a transcript section."""
        self.steps.append([])
        self.say(f"# {title}", MUTED)

    def say(self, text: str, color: tuple[int, int, int] = TEXT) -> None:
        """Print one transcript line and keep it for the GIF."""
        print(text)
        self.steps[-1].append((text, color))

    def chat(self, **body: object) -> httpx.Response:
        """Send one chat completion with the weather tool."""
        payload = {"tools": [WEATHER_TOOL], **body}
        return self._client.post("/v1/chat/completions", json=payload, headers=HEADERS)


def _ok(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON body of a successful response."""
    if response.status_code != 200:
        raise DemoError(f"HTTP {response.status_code}: {response.text}")
    body: dict[str, Any] = response.json()
    return body


def _sse_frames(body: str) -> list[str]:
    """Return the ``data:`` payloads of an SSE body."""
    return [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]


def run(session: Session) -> None:
    """Run the round trip, the streamed turn, and the rejected orphan result."""
    session.step("1) POST /v1/chat/completions  tools=[get_weather]  (the client offers a tool)")
    first = _ok(session.chat(messages=[QUESTION]))
    choice, router = first["choices"][0], first["router"]
    message = choice["message"]
    if not message.get("tool_calls"):
        raise DemoError("expected mock providers; start the server with scripts/serve_mock.py")
    [call] = message["tool_calls"]
    session.say(
        f"-> {router['model_used']}  finish_reason={choice['finish_reason']}  "
        f"content={json.dumps(message['content'])}",
        SUCCESS,
    )
    session.say(f"   tool_calls[0].id = {call['id']}")
    session.say(f"   {call['function']['name']}({call['function']['arguments']})", ACCENT)

    session.step("2) the client runs the tool itself (nexus forwards calls, never executes them)")
    arguments = json.loads(call["function"]["arguments"])
    result = get_weather(**arguments)
    session.say(f"get_weather(**{json.dumps(arguments)}) -> {json.dumps(result)}", ACCENT)

    session.step("3) POST /v1/chat/completions  + assistant tool_calls + role=tool result")
    tool_message = {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
    second = _ok(session.chat(messages=[QUESTION, message, tool_message]))
    final = second["choices"][0]
    session.say(
        f"-> {second['router']['model_used']}  finish_reason={final['finish_reason']}", SUCCESS
    )
    session.say(f"   content: {final['message']['content']}")

    session.step("4) the same first turn with stream=true: OpenAI chunks with delta.tool_calls")
    streamed = session.chat(messages=[QUESTION], stream=True)
    if streamed.status_code != 200:
        raise DemoError(f"stream failed: HTTP {streamed.status_code}")
    for frame in _sse_frames(streamed.text):
        if frame == "[DONE]":
            session.say("data: [DONE]", MUTED)
            continue
        chunk = json.loads(frame)["choices"][0]
        shown: dict[str, Any] = {"delta": chunk["delta"]}
        if chunk["finish_reason"] is not None:
            shown["finish_reason"] = chunk["finish_reason"]
        session.say(f"data: {json.dumps(shown)}")

    session.step("5) a tool result that answers no tool call is rejected before routing")
    orphan = session.chat(messages=[QUESTION, tool_message])
    error = orphan.json().get("error", {})
    session.say(
        f"-> HTTP {orphan.status_code} {error.get('type')}  param={error.get('param')}", WARN
    )
    session.say(f"   {error.get('message')}")


def main_cli(argv: Sequence[str] | None = None) -> int:
    """Parse flags, run the session, and optionally render the GIF."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--gif", type=Path, help="write the transcript as a GIF to this path")
    args = parser.parse_args(argv)
    session = Session(args.base_url)
    try:
        run(session)
    except (DemoError, httpx.HTTPError) as error:
        print(f"demo failed: {error}", file=sys.stderr)
        return 1
    if args.gif is not None:
        flags = " ".join(sys.argv[1:] if argv is None else argv)
        command = f"PYTHONPATH=src python scripts/tool_calling_demo.py {flags}"
        render_gif(args.gif, command, args.base_url, session.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
