"""Regression tests: catalog-priced models that adapters cannot price are not billed $0.

CONFIGURATION.md tells operators to extend the model catalog when onboarding
provider-specific SKUs (fine-tunes, newly released models). Each provider adapter
keeps its own price table, and ``estimate_cost`` returned 0.0 for any model
missing from it, so completions from such a model were recorded at $0 in the
response, spend ledger (``/v1/spend``), per-user budget, audit log, and
``router_cost_usd_total``, and budget caps never stopped further requests.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from structlog.testing import capture_logs

import api.main as main
from adapters.openai import OpenAIAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings, default_model_catalog
from router.engine import NexusRouter
from router.model_ids import GPT6_LONG_PROMPT_THRESHOLD_TOKENS, OPENAI_BALANCED_MODEL
from router.schemas import ChatMessage, DomainTag, LongPromptPricing, ModelCandidate
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.spend_ledger import SpendLedger
from safety.tenant_rate_limiter import TenantRateLimiter

# A customer fine-tune: no adapter price table can ever list it.
FINE_TUNE = "ft:gpt-6.1-terra:acme:support:7p4lkq2c"
FINE_TUNE_RATES = (0.003, 0.012)  # USD per 1K tokens (input, output)
# The same SKU family with placeholder $0 catalog prices: no price is known anywhere.
UNPRICED = "ft:gpt-6.1-terra:acme:triage:9d1x0b7e"
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _fine_tune(model: str, input_cost_per_1k: float, output_cost_per_1k: float) -> ModelCandidate:
    """Return an operator catalog row for a fine-tune served by the OpenAI adapter."""
    return ModelCandidate(
        model=model,
        provider="openai",
        quality_score=0.9,
        input_cost_per_1k=input_cost_per_1k,
        output_cost_per_1k=output_cost_per_1k,
        supports_domains={DomainTag.GENERAL, DomainTag.CODE},
        context_window=1_050_000,
        long_prompt_pricing=LongPromptPricing(
            threshold_tokens=GPT6_LONG_PROMPT_THRESHOLD_TOKENS,
            input_cost_per_1k=input_cost_per_1k * 2,
            output_cost_per_1k=output_cost_per_1k * 1.5,
        ),
    )


class _UsagelessStreamAdapter(OpenAIAdapter):
    """OpenAI adapter whose stream ends without a usage report (estimated accounting)."""

    async def stream(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> AsyncIterator[str]:
        response = await self.complete(model, messages, max_tokens)
        yield response.content


@dataclass
class _OpenAIStub:
    """In-process OpenAI chat-completions endpoint that reports fixed token usage."""

    prompt_tokens: int
    completion_tokens: int
    models: list[str] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        self.models.append(model)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": f"[{model}] ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                },
            },
        )


@dataclass
class _Gateway:
    """API client wired to a real OpenAI adapter and an in-process provider."""

    client: TestClient
    provider: _OpenAIStub
    audit_path: Path

    def chat(self, **body: object) -> httpx.Response:
        payload = {"messages": [{"role": "user", "content": "Draft a reply."}], "user": "alice"}
        return self.client.post(
            "/v1/chat/completions",
            json=payload | body,
            headers={"X-Router-Strategy": "failover-priority"},
        )

    def audit_records(self) -> list[dict[str, Any]]:
        lines = self.audit_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., _Gateway]:
    """Build a gateway whose catalog is the default one plus operator rows."""

    def build(
        *,
        extra: list[ModelCandidate],
        failover: list[str],
        usage: tuple[int, int] = (10_000, 2_000),
        budget_cap_usd: float = 25.0,
        adapter_type: type[OpenAIAdapter] = OpenAIAdapter,
    ) -> _Gateway:
        settings = RouterSettings(
            audit_log_path=str(tmp_path / "audit-log.jsonl"),
            spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
            idempotency_path=str(tmp_path / "idempotency.sqlite3"),
            budget_cap_usd=budget_cap_usd,
            failover_priority=failover,
        )
        catalog = default_model_catalog() | {candidate.model: candidate for candidate in extra}
        registry = AdapterRegistry({"openai": adapter_type("test-key", 5.0)})
        router = NexusRouter(settings, registry, model_catalog=catalog)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=60, refill_per_second=1.0),
        )
        provider = _OpenAIStub(*usage)

        def provider_client(*, timeout: float) -> httpx.AsyncClient:
            transport = httpx.MockTransport(provider.handle)
            return _REAL_ASYNC_CLIENT(transport=transport, timeout=timeout)

        monkeypatch.setattr("adapters.openai.httpx.AsyncClient", provider_client)
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: ResponseCache(enabled=False))
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        monkeypatch.setattr(
            main, "get_spend_ledger", lambda: SpendLedger(settings.spend_ledger_path)
        )
        return _Gateway(TestClient(main.app), provider, Path(settings.audit_log_path))

    return build


def _metered_cost_usd(model: str) -> float:
    """Return ``router_cost_usd_total`` for an OpenAI model (0.0 before its first sample)."""
    value = REGISTRY.get_sample_value(
        "router_cost_usd_total", {"provider": "openai", "model": model}
    )
    return value or 0.0


def _catalog_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Return the fine-tune's standard-rate catalog cost for the given usage."""
    input_rate, output_rate = FINE_TUNE_RATES
    return (input_tokens / 1000.0) * input_rate + (output_tokens / 1000.0) * output_rate


@pytest.mark.parametrize(
    ("usage", "expected_cost_usd"),
    [
        pytest.param((10_000, 2_000), 0.054, id="standard-rates"),
        # Above 272K input tokens the catalog's long-prompt tier bills 2x / 1.5x.
        pytest.param((300_000, 2_000), 1.836, id="long-prompt-tier"),
    ],
)
def test_catalog_priced_model_is_billed_at_catalog_price(
    gateway: Callable[..., _Gateway], usage: tuple[int, int], expected_cost_usd: float
) -> None:
    """Every spend sink records the catalog price, not $0, for a model adapters cannot price."""
    gw = gateway(extra=[_fine_tune(FINE_TUNE, *FINE_TUNE_RATES)], failover=[FINE_TUNE], usage=usage)
    metered_before = _metered_cost_usd(FINE_TUNE)

    response = gw.chat()

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == FINE_TUNE
    assert body["router"]["cost_usd"] == pytest.approx(expected_cost_usd)
    spend = gw.client.get("/v1/spend").json()
    assert spend["by_model"] == {FINE_TUNE: pytest.approx(expected_cost_usd)}
    assert spend["by_tenant"] == {"alice": pytest.approx(expected_cost_usd)}
    (audit,) = gw.audit_records()
    assert audit["token_cost"] == pytest.approx(expected_cost_usd)
    assert _metered_cost_usd(FINE_TUNE) - metered_before == pytest.approx(expected_cost_usd)


def test_catalog_priced_spend_counts_toward_the_budget_cap(
    gateway: Callable[..., _Gateway],
) -> None:
    """Spend recorded at $0 never consumed the budget, so the cap never stopped requests."""
    gw = gateway(
        extra=[_fine_tune(FINE_TUNE, *FINE_TUNE_RATES)], failover=[FINE_TUNE], budget_cap_usd=0.05
    )

    first = gw.chat()
    second = gw.chat()

    assert first.status_code == 200
    assert first.json()["router"]["cost_usd"] == pytest.approx(0.054)
    assert second.status_code != 200
    assert "budget cap exceeded for alice" in second.text
    assert gw.provider.models == [FINE_TUNE]


@pytest.mark.parametrize(
    "adapter_type",
    [
        pytest.param(OpenAIAdapter, id="stream-reports-usage"),
        pytest.param(_UsagelessStreamAdapter, id="stream-without-usage"),
    ],
)
def test_streamed_catalog_priced_completion_is_billed(
    gateway: Callable[..., _Gateway], adapter_type: type[OpenAIAdapter]
) -> None:
    """Streams are billed at catalog rates whether or not the provider reports usage."""
    gw = gateway(
        extra=[_fine_tune(FINE_TUNE, *FINE_TUNE_RATES)],
        failover=[FINE_TUNE],
        adapter_type=adapter_type,
    )

    response = gw.chat(stream=True)

    assert response.status_code == 200
    assert f"[{FINE_TUNE}] ok" in response.text
    assert '"error"' not in response.text
    (audit,) = gw.audit_records()
    expected_cost_usd = _catalog_cost_usd(audit["input_tokens"], audit["output_tokens"])
    assert expected_cost_usd > 0
    assert audit["token_cost"] == pytest.approx(expected_cost_usd)
    spend = gw.client.get("/v1/spend").json()
    assert spend["by_model"] == {FINE_TUNE: pytest.approx(expected_cost_usd)}


def test_model_with_no_known_price_is_never_dispatched(gateway: Callable[..., _Gateway]) -> None:
    """A model neither the adapter nor the catalog prices fails closed and warns once.

    Its spend could not be recorded or budgeted, so the engine skips it and routes
    to the next priced model instead of billing $0.
    """
    with capture_logs() as logs:
        gw = gateway(
            extra=[_fine_tune(UNPRICED, 0.0, 0.0)], failover=[UNPRICED, OPENAI_BALANCED_MODEL]
        )
        responses = [gw.chat(), gw.chat()]

    assert [response.status_code for response in responses] == [200, 200]
    assert [response.json()["model"] for response in responses] == [OPENAI_BALANCED_MODEL] * 2
    assert all(response.json()["router"]["cost_usd"] > 0 for response in responses)
    assert gw.provider.models == [OPENAI_BALANCED_MODEL] * 2
    unknown_price_warnings = [
        entry
        for entry in logs
        if entry["event"] == "model_price_unknown" and entry["log_level"] == "warning"
    ]
    assert [entry["model"] for entry in unknown_price_warnings] == [UNPRICED]
