"""Regression tests: scheduled list-price changes take effect on their date.

Google raises Gemini 4 Carbon and Gemini 4 Argon from $2 / $10 to $4 / $20 per 1M
tokens on 2026-12-01, and Gemini 3.8 Flash from $0.75 / $3.75 to $1.50 / $7.50 on
2027-01-01 (Gemini API pricing page, verified 2026-11-27). Nexus priced every model
from static tables, so from those dates a deployment that was not edited and
redeployed at the right moment billed these models at the old prices in the
response, spend ledger, and audit log, admitted requests its budget cap should
refuse, and advertised the old prices in ``GET /v1/models``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as main
import router.config as config
from adapters.anthropic import AnthropicAdapter
from adapters.google import GoogleGeminiAdapter
from adapters.mock import MockProviderAdapter
from adapters.moonshot import MoonshotAdapter
from adapters.openai import OpenAIAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings, default_model_catalog
from router.engine import NexusRouter
from router.schemas import (
    ChatMessage,
    DomainTag,
    ModelCandidate,
    RouterRequest,
    RoutingStrategyName,
)
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.spend_ledger import SpendLedger
from safety.tenant_rate_limiter import TenantRateLimiter

CARBON = "gemini-4-carbon"
ARGON = "gemini-4-argon"
FLASH = "gemini-3.8-flash"
NEON = "gemini-4-neon"
DEC_1 = datetime(2026, 12, 1, tzinfo=UTC)
JAN_1 = datetime(2027, 1, 1, tzinfo=UTC)
LAST_SECOND_OF_NOV_30 = datetime(2026, 11, 30, 23, 59, 59, tzinfo=UTC)
LAST_SECOND_OF_DEC_31 = datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC)
# Token usage the in-process Gemini endpoint reports for every completion.
PROMPT_TOKENS = 100_000
OUTPUT_TOKENS = 20_000
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _cost(input_per_1m: float, output_per_1m: float) -> float:
    """Return the cost of the reported usage at per-1M-token list prices."""
    return (PROMPT_TOKENS * input_per_1m + OUTPUT_TOKENS * output_per_1m) / 1_000_000


def _price_at(monkeypatch: pytest.MonkeyPatch, at: datetime) -> None:
    """Pin the default pricing clock, ``router.config.utc_now``, to ``at``.

    ``utc_now`` reads ``datetime`` from its module's globals. ``raising=False``
    creates the attribute if it is missing, so a build that ignores the clock
    fails these tests on the stale price rather than on the patch.
    """

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:  # type: ignore[override]
            return at if tz is None else at.astimezone(tz)

    monkeypatch.setattr(config, "datetime", _FrozenDatetime, raising=False)


def _gemini_answer(request: httpx.Request) -> httpx.Response:
    """Answer a Gemini ``generateContent`` call with the fixed token usage."""
    del request
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": PROMPT_TOKENS,
                "candidatesTokenCount": OUTPUT_TOKENS,
            },
        },
    )


def _gemini_client(*, timeout: float) -> httpx.AsyncClient:
    """Return an HTTP client whose Gemini endpoint runs in-process."""
    return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(_gemini_answer), timeout=timeout)


def _carbon_row() -> ModelCandidate:
    """Return the catalog row an operator adds to route Gemini 4 Carbon."""
    return ModelCandidate(
        model=CARBON,
        provider="google",
        quality_score=0.96,
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.010,
        supports_domains={DomainTag.CODE, DomainTag.GENERAL},
        supports_realtime=False,
        context_window=1_048_576,
    )


@dataclass
class _Gateway:
    """API client routing every request to one model on the real Gemini adapter."""

    client: TestClient
    router: NexusRouter
    audit_path: Path

    def chat(self, *, max_tokens: int = 1_000, stream: bool = False) -> httpx.Response:
        return self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Draft a reply."}],
                "max_tokens": max_tokens,
                "user": "alice",
                "stream": stream,
            },
            headers={"X-Router-Strategy": "failover-priority"},
        )

    def listed_prices(self) -> dict[str, tuple[float, float]]:
        """Return ``GET /v1/models`` per-1M (input, output) prices by model id."""
        models = self.client.get("/v1/models").json()["data"]
        return {
            model["id"]: (
                model["nexus"]["pricing"]["input_per_1m_tokens"],
                model["nexus"]["pricing"]["output_per_1m_tokens"],
            )
            for model in models
        }

    def audit_costs(self) -> list[float]:
        lines = self.audit_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line)["token_cost"] for line in lines]


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., _Gateway]:
    """Build a gateway over the default catalog plus Carbon that routes only ``model``."""

    def build(
        model: str,
        *,
        budget_cap_usd: float = 25.0,
        clock: Callable[[], datetime] | None = None,
    ) -> _Gateway:
        settings = RouterSettings(
            audit_log_path=str(tmp_path / "audit-log.jsonl"),
            spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
            idempotency_path=str(tmp_path / "idempotency.sqlite3"),
            feedback_path=str(tmp_path / "feedback.sqlite3"),
            budget_cap_usd=budget_cap_usd,
            failover_priority=[model],
        )
        catalog = default_model_catalog() | {CARBON: _carbon_row()}
        registry = AdapterRegistry({"google": GoogleGeminiAdapter("test-key", 5.0)})
        router = NexusRouter(settings, registry, model_catalog=catalog, clock=clock)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=60, refill_per_second=1.0),
        )
        monkeypatch.setattr("adapters.google.httpx.AsyncClient", _gemini_client)
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: ResponseCache(enabled=False))
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        monkeypatch.setattr(
            main, "get_spend_ledger", lambda: SpendLedger(settings.spend_ledger_path)
        )
        return _Gateway(TestClient(main.app), router, Path(settings.audit_log_path))

    return build


@pytest.mark.parametrize("model", [CARBON, ARGON])
@pytest.mark.parametrize(
    ("at", "list_prices"),
    [
        pytest.param(LAST_SECOND_OF_NOV_30, (2.0, 10.0), id="2026-11-30T23:59:59Z"),
        pytest.param(DEC_1, (4.0, 20.0), id="2026-12-01T00:00:00Z"),
    ],
)
def test_gemini_4_carbon_and_argon_bill_the_price_in_effect(
    gateway: Callable[..., _Gateway],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    at: datetime,
    list_prices: tuple[float, float],
) -> None:
    """Carbon and Argon completions bill $4 / $20 per 1M tokens from 2026-12-01T00:00:00Z."""
    _price_at(monkeypatch, at)
    gw = gateway(model)

    response = gw.chat()

    assert response.status_code == 200
    expected_cost_usd = _cost(*list_prices)
    assert response.json()["router"]["cost_usd"] == pytest.approx(expected_cost_usd)
    spend = gw.client.get("/v1/spend").json()
    assert spend["by_model"] == {model: pytest.approx(expected_cost_usd)}
    assert gw.audit_costs() == [pytest.approx(expected_cost_usd)]


@pytest.mark.parametrize(
    ("at", "list_prices"),
    [
        pytest.param(LAST_SECOND_OF_DEC_31, (0.75, 3.75), id="2026-12-31T23:59:59Z"),
        pytest.param(JAN_1, (1.5, 7.5), id="2027-01-01T00:00:00Z"),
    ],
)
def test_gemini_3_8_flash_bills_the_price_in_effect(
    gateway: Callable[..., _Gateway],
    monkeypatch: pytest.MonkeyPatch,
    at: datetime,
    list_prices: tuple[float, float],
) -> None:
    """Gemini 3.8 Flash completions bill $1.50 / $7.50 per 1M tokens from 2027-01-01."""
    _price_at(monkeypatch, at)
    gw = gateway(FLASH)

    response = gw.chat()

    assert response.status_code == 200
    assert response.json()["router"]["cost_usd"] == pytest.approx(_cost(*list_prices))


async def test_gemini_adapter_bills_carbon_at_the_price_in_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Gemini adapter's own completion cost uses the new price from 2026-12-01."""
    _price_at(monkeypatch, DEC_1)
    monkeypatch.setattr("adapters.google.httpx.AsyncClient", _gemini_client)
    adapter = GoogleGeminiAdapter("test-key", 5.0)

    response = await adapter.complete(CARBON, [ChatMessage(role="user", content="Hi.")], 100)

    assert response.cost_usd == pytest.approx(_cost(4.0, 20.0))


@pytest.mark.parametrize(
    ("at", "model", "list_prices"),
    [
        pytest.param(DEC_1, ARGON, (4.0, 20.0), id="argon-2026-12-01"),
        pytest.param(JAN_1, FLASH, (1.5, 7.5), id="flash-2027-01-01"),
    ],
)
def test_catalog_and_models_api_agree_with_billing(
    gateway: Callable[..., _Gateway],
    monkeypatch: pytest.MonkeyPatch,
    at: datetime,
    model: str,
    list_prices: tuple[float, float],
) -> None:
    """At one instant the catalog, ``GET /v1/models``, and billing use the same price."""
    _price_at(monkeypatch, at)
    gw = gateway(model)

    assert gw.listed_prices()[model] == pytest.approx(list_prices)
    pricing = gw.client.get(f"/v1/models/{model}").json()["nexus"]["pricing"]
    assert (pricing["input_per_1m_tokens"], pricing["output_per_1m_tokens"]) == list_prices
    candidate = gw.router.model_catalog[model]
    catalog_prices = (candidate.input_cost_per_1k * 1000, candidate.output_cost_per_1k * 1000)
    assert catalog_prices == pytest.approx(list_prices)
    assert gw.chat().json()["router"]["cost_usd"] == pytest.approx(_cost(*list_prices))


@pytest.mark.parametrize(
    ("at", "status_code"),
    [
        pytest.param(LAST_SECOND_OF_NOV_30, 200, id="2026-11-30T23:59:59Z"),
        pytest.param(DEC_1, 402, id="2026-12-01T00:00:00Z"),
    ],
)
def test_budget_estimate_uses_the_price_in_effect(
    gateway: Callable[..., _Gateway],
    monkeypatch: pytest.MonkeyPatch,
    at: datetime,
    status_code: int,
) -> None:
    """The pre-dispatch budget estimate prices Argon at the rate billing will charge.

    The estimate covers ``max_tokens`` of output: 10,000 tokens cost $0.10 at the
    old $10 per 1M output rate, within a $0.15 cap, and $0.20 at the new $20.
    """
    _price_at(monkeypatch, at)
    gw = gateway(ARGON, budget_cap_usd=0.15)

    response = gw.chat(max_tokens=10_000)

    assert response.status_code == status_code
    if status_code == 402:
        assert response.json()["detail"] == "budget cap exceeded for alice"


def test_models_without_a_scheduled_change_keep_their_prices(
    gateway: Callable[..., _Gateway], monkeypatch: pytest.MonkeyPatch
) -> None:
    """After every scheduled change, all other models still list and bill their prices."""
    _price_at(monkeypatch, datetime(2027, 6, 1, tzinfo=UTC))
    gw = gateway(NEON)

    listed = gw.listed_prices()
    for model, candidate in default_model_catalog().items():
        if model in {ARGON, FLASH}:
            continue
        static = (candidate.input_cost_per_1k * 1000, candidate.output_cost_per_1k * 1000)
        assert listed[model] == pytest.approx(static), model
    assert gw.chat().json()["router"]["cost_usd"] == pytest.approx(_cost(3.0, 15.0))
    adapters_and_prices: list[tuple[Any, str, float]] = [
        (OpenAIAdapter("k", 5.0), "gpt-6.2-astra", 0.048),
        (AnthropicAdapter("k", 5.0), "claude-opus-5-5", 0.024),
        (GoogleGeminiAdapter("k", 5.0), "gemini-4-hydrogen", 0.009),
        (MoonshotAdapter("k", "https://api.moonshot.ai/v1", 5.0), "kimi-k3.1-pro", 0.012),
    ]
    for adapter, model, cost_of_1k_in_1k_out in adapters_and_prices:
        assert adapter.estimate_cost(model, 1_000, 1_000) == pytest.approx(cost_of_1k_in_1k_out)


@pytest.mark.parametrize("stream", [False, True], ids=["complete", "stream"])
def test_a_request_routed_before_a_change_keeps_its_price(
    gateway: Callable[..., _Gateway], monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """A request's price is fixed when it is routed, even if it completes after the change.

    The router routes at 23:59:59 UTC; the Gemini adapter prices the answer when
    it arrives, after midnight. The request bills the old price; the next one,
    routed at midnight, bills the new price.
    """
    _price_at(monkeypatch, DEC_1)
    routed_at = [LAST_SECOND_OF_NOV_30]
    gw = gateway(ARGON, clock=lambda: routed_at[0])

    assert gw.chat(stream=stream).status_code == 200
    routed_at[0] = DEC_1
    assert gw.chat(stream=stream).status_code == 200

    expected = [pytest.approx(_cost(2.0, 10.0)), pytest.approx(_cost(4.0, 20.0))]
    assert gw.audit_costs() == expected
    spend = gw.client.get("/v1/spend").json()
    assert spend["by_model"] == {ARGON: pytest.approx(_cost(2.0, 10.0) + _cost(4.0, 20.0))}


@pytest.mark.parametrize("stream", [False, True])
def test_tool_calling_requests_bill_the_price_they_were_routed_at(
    gateway: Callable[..., _Gateway], monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """Tool requests (streamed ones make one non-streaming call) keep the routed price."""
    _price_at(monkeypatch, DEC_1)
    routed_at = [LAST_SECOND_OF_NOV_30]
    gw = gateway(ARGON, clock=lambda: routed_at[0])
    tool = {
        "type": "function",
        "function": {"name": "lookup_order", "parameters": {"type": "object", "properties": {}}},
    }
    body = {
        "messages": [{"role": "user", "content": "Draft a reply."}],
        "max_tokens": 1_000,
        "user": "alice",
        "stream": stream,
        "tools": [tool],
    }
    headers = {"X-Router-Strategy": "failover-priority"}

    assert gw.client.post("/v1/chat/completions", json=body, headers=headers).status_code == 200
    routed_at[0] = DEC_1
    assert gw.client.post("/v1/chat/completions", json=body, headers=headers).status_code == 200

    assert gw.audit_costs() == [pytest.approx(_cost(2.0, 10.0)), pytest.approx(_cost(4.0, 20.0))]


def test_a_naive_clock_is_rejected(gateway: Callable[..., _Gateway]) -> None:
    """Prices are compared with UTC instants, so the clock must be timezone-aware."""
    with pytest.raises(ValueError, match="timezone-aware"):
        gateway(ARGON, clock=lambda: datetime(2026, 12, 1))


async def test_a_running_router_ranks_models_at_the_price_in_effect(tmp_path: Path) -> None:
    """Cost-aware routing on a running router follows a change once it takes effect.

    Argon ($2 / $10 per 1M) undercuts Neon ($3 / $15) until 2026-12-01 and costs
    more ($4 / $20) from then on, so ``cost-optimal`` switches to Neon.
    """
    now = [LAST_SECOND_OF_NOV_30]
    catalog = {
        model: default_model_catalog()[model].model_copy(update={"supports_realtime": True})
        for model in (ARGON, NEON)
    }
    settings = RouterSettings(
        audit_log_path=str(tmp_path / "audit-log.jsonl"),
        spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
        feedback_path=str(tmp_path / "feedback.sqlite3"),
        ab_model_a=ARGON,
        ab_model_b=NEON,
        canary_stable_model=ARGON,
        canary_model=NEON,
        failover_priority=[ARGON, NEON],
    )
    registry = AdapterRegistry({"google": MockProviderAdapter("google")})
    router = NexusRouter(settings, registry, model_catalog=catalog, clock=lambda: now[0])

    async def cost_optimal_pick(request_id: str) -> str:
        request = RouterRequest(
            request_id=request_id,
            messages=[ChatMessage(role="user", content="Draft a reply.")],
            strategy=RoutingStrategyName.COST_OPTIMAL,
        )
        return (await router.complete(request)).model_used

    assert await cost_optimal_pick("before") == ARGON
    now[0] = DEC_1
    assert await cost_optimal_pick("after") == NEON
