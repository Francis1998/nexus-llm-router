"""Run a local routing demo with deterministic mock providers."""

import asyncio
from pathlib import Path

from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from observability.logging import configure_logging
from router.config import RouterSettings
from router.engine import NexusRouter
from router.schemas import ChatMessage, RouterRequest, RoutingStrategyName


def build_demo_router() -> NexusRouter:
    """Build an offline demo router.

    Returns:
        Router configured with mock providers.
    """
    return NexusRouter(
        settings=RouterSettings(audit_log_path="migrations/demo-audit.jsonl"),
        adapter_registry=AdapterRegistry(
            {
                "openai": MockProviderAdapter("openai"),
                "anthropic": MockProviderAdapter("anthropic"),
                "google": MockProviderAdapter("google"),
                "moonshot": MockProviderAdapter("moonshot"),
            },
        ),
    )


async def run_demo() -> None:
    """Run demo prompts through different routing strategies."""
    configure_logging()
    Path("migrations").mkdir(exist_ok=True)
    router = build_demo_router()
    prompts = [
        ("medical", "Assess patient symptoms and treatment risk.", RoutingStrategyName.RULE_BASED),
        (
            "code",
            "Debug this Python async function and propose tests.",
            RoutingStrategyName.CLASSIFIER,
        ),
        ("cost", "Summarize this rollout update for executives.", RoutingStrategyName.COST_OPTIMAL),
        ("ab", "Write a concise release note.", RoutingStrategyName.AB_TEST),
    ]
    for request_id, prompt, strategy in prompts:
        response = await router.complete(
            RouterRequest(
                request_id=f"demo-{request_id}",
                messages=[ChatMessage(content=prompt)],
                strategy=strategy,
            ),
        )
        print(
            f"{request_id}: strategy={response.routing_strategy.value} "
            f"model={response.model_used} cost=${response.cost_usd:.6f} "
            f"rationale={response.rationale}",
        )


if __name__ == "__main__":
    asyncio.run(run_demo())
