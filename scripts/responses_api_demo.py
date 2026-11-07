"""Call nexus ``POST /v1/responses`` with the official OpenAI Python SDK.

Start a server first (``PYTHONPATH=src python scripts/serve_mock.py``), then::

    uv run --with openai python scripts/responses_api_demo.py --base-url http://127.0.0.1:8000/v1
"""

import argparse

from openai import BadRequestError, OpenAI


def run(base_url: str) -> None:
    """Exercise create, raw streaming, the stream helper, and an unsupported field."""
    client = OpenAI(base_url=base_url, api_key="sk-demo-local")

    response = client.responses.create(
        input="Summarize our refund policy in one line.",
        instructions="You are a terse support agent.",
        metadata={"ticket": "T-42"},
        extra_headers={"X-Router-Strategy": "cost-optimal"},
    )
    nexus = (response.model_extra or {}).get("nexus", {})
    usage = response.usage
    print(f"create: {response.id[:13]}... {response.status} model={response.model}")
    print(f"  output_text={response.output_text!r} metadata={response.metadata}")
    print(
        f"  usage in={usage.input_tokens} out={usage.output_tokens} total={usage.total_tokens}"
        f" | nexus strategy={nexus['routing_strategy']} cost_usd={nexus['cost_usd']:.6f}"
    )

    events = list(client.responses.create(input="Stream a status update.", stream=True))
    labels = [f"{event.sequence_number}:{event.type}" for event in events]
    print("stream=True events (sequence_number:type):")
    for start in range(0, len(labels), 3):
        print("  " + " ".join(labels[start : start + 3]))
    deltas = [event.delta for event in events if event.type == "response.output_text.delta"]
    print(f"  deltas={deltas}")

    # The SDK stream helper requires a model; ids outside the nexus catalog let
    # the routing strategy choose.
    with client.responses.stream(model="auto", input="Stream a status update.") as stream:
        final = stream.get_final_response()
    print(f"responses.stream(): {final.output_text!r} total_tokens={final.usage.total_tokens}")

    try:
        client.responses.create(input="Continue.", previous_response_id="resp_123")
    except BadRequestError as error:
        print(f"unsupported: HTTP {error.status_code} code={error.code} param={error.param}")


def main() -> None:
    """Parse CLI flags and run the demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run(parser.parse_args().base_url)


if __name__ == "__main__":
    main()
