"""Call nexus ``POST /v1/messages`` with the official Anthropic Python SDK.

Start a server first (``PYTHONPATH=src python scripts/serve_mock.py``), then::

    uv run --with anthropic python scripts/anthropic_messages_demo.py --base-url http://127.0.0.1:8000

The SDK appends ``/v1/messages`` to ``base_url``, so pass the server root.
"""

import argparse

import anthropic

PROMPT = (
    "Design a multi-region failover strategy for a payments platform: compare "
    "active-active, active-passive, and pilot-light architectures, analyze cost, "
    "latency, and consistency trade-offs, and recommend one with a phased rollout plan."
)
WEATHER_TOOL = {"name": "get_weather", "input_schema": {"type": "object"}}


def run(base_url: str) -> None:
    """Exercise create, the stream helper, count_tokens, stop sequences, and a tool call."""
    client = anthropic.Anthropic(base_url=base_url, api_key="sk-demo-local")
    messages = [{"role": "user", "content": PROMPT}]

    message = client.messages.create(
        model="claude-opus-5-5", max_tokens=256, system="Be terse.", messages=messages
    )
    nexus = (message.model_extra or {})["nexus"]
    usage = message.usage
    print(f"create: {message.id[:12]}... model={message.model} stop={message.stop_reason}")
    print(f"  text={message.content[0].text!r} usage in={usage.input_tokens} out={usage.output_tokens}")
    print(f"  nexus strategy={nexus['routing_strategy']} rationale={nexus['rationale']!r}")

    with client.messages.stream(model="gpt-6-astra", max_tokens=256, messages=messages) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    print(f"stream: text_stream={text!r} stop={final.stop_reason} out={final.usage.output_tokens}")

    count = client.messages.count_tokens(
        model="claude-opus-5-5", system="Be terse.", messages=messages
    )
    print(f"count_tokens: input_tokens={count.input_tokens} (router estimate, no provider call)")

    stopped = client.messages.create(
        model="claude-opus-5-5", max_tokens=256, messages=messages, stop_sequences=["routed"]
    )
    print(
        f"stop_sequences: text={stopped.content[0].text!r} stop={stopped.stop_reason} "
        f"stop_sequence={stopped.stop_sequence!r}"
    )

    try:
        client.messages.create(
            model="claude-opus-5-5", max_tokens=256, messages=messages, tools=[WEATHER_TOOL]
        )
    except anthropic.BadRequestError as error:
        print(f"tools: HTTP {error.status_code} {error.body['error']['type']}")
    print(f"anthropic {anthropic.__version__}")


def main() -> None:
    """Parse CLI flags and run the demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    run(parser.parse_args().base_url)


if __name__ == "__main__":
    main()
