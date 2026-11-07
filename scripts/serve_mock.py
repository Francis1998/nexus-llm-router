"""Serve the Nexus API with deterministic mock providers (no API keys needed).

Reproduces the docs demos offline and lets OpenAI SDK clients exercise the
real HTTP surface (``/v1/chat/completions``, ``/v1/responses``, ...)::

    PYTHONPATH=src python scripts/serve_mock.py --port 8000
"""

import argparse

import uvicorn

import api.main as main
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry

PROVIDERS = ("openai", "anthropic", "google", "moonshot")


def install_mock_providers(output_tokens: int = 16) -> None:
    """Route every provider to a deterministic mock adapter.

    Args:
        output_tokens: Mock answer length in tokens; requests with a smaller
            ``max_tokens`` / ``max_output_tokens`` are reported as truncated.
    """
    registry = AdapterRegistry(
        {name: MockProviderAdapter(name, output_tokens=output_tokens) for name in PROVIDERS}
    )
    main.get_router.cache_clear()
    main.get_adapter_registry = lambda: registry


def main_cli() -> None:
    """Parse CLI flags and run uvicorn with mock providers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--mock-output-tokens",
        type=int,
        default=16,
        help="mock answer length in tokens; smaller output caps report truncation",
    )
    args = parser.parse_args()
    install_mock_providers(args.mock_output_tokens)
    uvicorn.run(main.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main_cli()
