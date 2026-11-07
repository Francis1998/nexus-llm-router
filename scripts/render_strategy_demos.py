"""Render strategy-guide demo GIFs from real, in-process routing decisions.

Each demo runs the actual FastAPI app (``api.main.app``) in-process over an
ASGI transport with deterministic mock provider adapters -- no network, no API
keys -- and records two requests for one routing strategy: one *without* the
signal its guide documents and one *with* it. The GIF shows both requests
(abridged) and the decision the router actually returned for each (served
model, strategy, rationale); none of that output is typed by hand.

Signals reach the router the way the code accepts them today:

* runtime state (latency, failures, 429 cool-downs, spend and token windows,
  in-flight depth) is produced by real warm-up requests through the app;
* request ``metadata`` and ``region`` are set on ``RouterRequest`` and routed by
  the same ``NexusRouter`` in-process, because the HTTP endpoints do not forward
  request-body ``metadata`` (or a ``region``) to routing strategies yet. Every
  render re-checks that gap -- the same metadata sent over HTTP must still route
  like the no-signal request -- and fails once it is closed, so the note printed
  in the GIFs cannot go stale.

Request ids are seeded so hash-based slices (sticky rings, canary buckets,
round-robin tie-breaks) render identically on every run; demos that rely on
mock latency show wall-clock measurements, which vary by a few milliseconds.

Usage::

    PYTHONPATH=src python scripts/render_strategy_demos.py                 # all demos
    PYTHONPATH=src python scripts/render_strategy_demos.py --only token-rpm-ceiling
    PYTHONPATH=src python scripts/render_strategy_demos.py --check         # print, write nothing
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from itertools import count
from pathlib import Path
from typing import Any

import httpx
from create_demo_gif import (
    ACCENT,
    BACKGROUND,
    HEIGHT,
    MUTED,
    PADDING,
    PROMPT,
    SUCCESS,
    TEXT,
    WARN,
    WIDTH,
    Line,
    save_gif,
)
from PIL import Image, ImageDraw, ImageFont

import api.main as main
from adapters.base import ProviderError
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings
from router.engine import NexusRouter
from router.schemas import ChatMessage, ProviderResponse, RouterRequest, RoutingStrategyName

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = REPO_ROOT / "build" / "strategy-demo-state"
PROVIDERS = ("openai", "anthropic", "google", "moonshot")
PROMPT_TEXT = "Summarize this customer call in three bullet points."
HTTP_GAP_NOTE = (
    "HTTP note: /v1/chat/completions, /v1/responses and /v1/messages do not forward body "
    "metadata/region to strategies yet (re-verified on every render)."
)
STATE_NOTE = (
    "Runtime state comes from real requests sent through the app first (mock providers, "
    "no network)."
)
HTTP_NOTE = "Both requests go through the real app over HTTP (mock providers, no network)."
FRAME_MS = 1200
LINE_HEIGHT = 19
PALETTE_COLORS = 24


class RetryAfterError(ProviderError):
    """Provider 429 carrying a ``Retry-After`` wait, as real SDK errors do."""

    def __init__(self, message: str, retry_after: float) -> None:
        """Initialize the error.

        Args:
            message: Human-readable failure description.
            retry_after: Seconds the provider asked callers to wait.
        """
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class DemoAdapter(MockProviderAdapter):
    """Mock provider with scripted latency, failures, 429s, or held calls."""

    def __init__(
        self,
        provider_name: str,
        *,
        delays_ms: Sequence[float] = (),
        failing: bool = False,
        retry_after: float | None = None,
        hold: asyncio.Event | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            provider_name: Provider name returned by the adapter.
            delays_ms: Per-call latency; the last value repeats once exhausted.
            failing: Whether calls raise an HTTP 503 provider error.
            retry_after: When set, calls raise HTTP 429 with this ``Retry-After``.
            hold: When set, calls block until the event is set (in-flight depth).
        """
        super().__init__(provider_name)
        self._delays_ms = list(delays_ms)
        self._failing = failing
        self._retry_after = retry_after
        self._hold = hold
        self.calls = 0

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        """Apply the scripted behaviour, then answer like the base mock."""
        delay_ms = (
            self._delays_ms[min(self.calls, len(self._delays_ms) - 1)] if self._delays_ms else 0
        )
        self.calls += 1
        if self._hold is not None:
            await self._hold.wait()
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000.0)
        if self._retry_after is not None:
            raise RetryAfterError(
                f"{self.provider_name} rate limited (HTTP 429)", self._retry_after
            )
        if self._failing:
            raise ProviderError(f"{self.provider_name} unavailable (HTTP 503)", status_code=503)
        return await super().complete(model, messages, max_tokens)


@dataclass(frozen=True)
class Probe:
    """One request shown in a demo panel, plus the real traffic sent before it."""

    metadata: Mapping[str, Any] | None = None
    region: str | None = None
    user: str | None = None
    warmup: int = 0
    warmup_strategy: str | None = None
    warmup_until_each_provider: int = 0
    failing: tuple[str, ...] = ()
    retry_after: Mapping[str, float] = field(default_factory=dict)
    delays_ms: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    held_in_flight: tuple[str, int] | None = None

    @property
    def in_process(self) -> bool:
        """Whether the probe needs ``RouterRequest`` fields HTTP does not forward."""
        return self.metadata is not None or self.region is not None


@dataclass(frozen=True)
class Scenario:
    """A strategy demo: the same strategy without and with its documented signal."""

    strategy: str
    with_signal: Probe
    without_signal: Probe = Probe()
    env: Mapping[str, str] = field(default_factory=dict)
    env_note: str = ""

    @property
    def guide_path(self) -> Path:
        """Repository path of the strategy guide."""
        return REPO_ROOT / "docs" / "guides" / f"{self.strategy.upper().replace('-', '_')}_GUIDE.md"

    @property
    def gif_path(self) -> Path:
        """Path of the demo GIF the guide embeds; the render replaces it in place.

        Raises:
            RuntimeError: If the guide does not embed a ``<strategy>.gif`` demo.
        """
        guide = self.guide_path.read_text(encoding="utf-8")
        embed = re.search(rf"!\[[^\]]*\]\(([^)\s]*\b{re.escape(self.strategy)}\.gif)\)", guide)
        if embed is None:
            raise RuntimeError(f"{self.guide_path} does not embed {self.strategy}.gif")
        return (self.guide_path.parent / embed.group(1)).resolve()


@dataclass(frozen=True)
class Outcome:
    """What the router returned for one probe."""

    status: int
    model: str
    provider: str
    strategy: str
    rationale: str
    warmup_served: tuple[str, ...] = ()


def capability(strategy: str, metadata: Mapping[str, Any]) -> Scenario:
    """Return a demo whose signal is request metadata."""
    return Scenario(strategy, with_signal=Probe(metadata=metadata))


SCENARIOS: tuple[Scenario, ...] = (
    capability(
        "audio-input-prefer",
        {
            "requires_audio": True,
            "audio_models": ["gemini-3.8-flash"],
            "model_capabilities": {"gemini-3.8-flash": "audio"},
        },
    ),
    capability("batch-api-prefer", {"requires_batch": True, "batch_models": ["gpt-6-luna"]}),
    Scenario(
        "carbon-aware-preference",
        with_signal=Probe(
            metadata={"carbon_intensity:anthropic": "450", "carbon_intensity:google": "120"}
        ),
        env={"NEXUS_CARBON_AWARE_MAX_INTENSITY": "400"},
    ),
    Scenario(
        "carbon-latency-blend",
        with_signal=Probe(
            metadata={
                "region": "eu",
                "carbon_intensity:anthropic:eu": "180",
                "carbon_intensity:openai:eu": "260",
            }
        ),
        env={
            "NEXUS_CARBON_LATENCY_CARBON_WEIGHT": "0.5",
            "NEXUS_CARBON_LATENCY_LATENCY_WEIGHT": "0.5",
        },
    ),
    capability(
        "code-execution-prefer",
        {"requires_code_execution": True, "code_execution_models": ["gemini-4-argon"]},
    ),
    capability(
        "computer-use-prefer",
        {
            "requires_computer_use": True,
            "computer_use_models": ["gemini-3.8-flash"],
            "model_capabilities": {"gemini-3.8-flash": "computer_use"},
        },
    ),
    Scenario(
        "cost-latency-pareto",
        with_signal=Probe(
            delays_ms={
                "openai": (300.0,),
                "anthropic": (90.0,),
                "google": (40.0,),
                "moonshot": (150.0,),
            },
            warmup_strategy="round-robin",
            warmup_until_each_provider=2,
        ),
    ),
    capability(
        "embeddings-prefer",
        {"requires_embeddings": True, "embeddings_models": ["gemini-3.8-flash"]},
    ),
    capability(
        "fine-tune-prefer", {"requires_fine_tune": True, "fine_tune_models": ["gpt-6-luna"]}
    ),
    Scenario(
        "health-cost-latency",
        with_signal=Probe(
            failing=("openai",),
            delays_ms={
                "openai": (300.0,),
                "anthropic": (90.0,),
                "google": (40.0,),
                "moonshot": (150.0,),
            },
            warmup_strategy="round-robin",
            warmup_until_each_provider=2,
        ),
        env={
            "NEXUS_HCL_HEALTH_WEIGHT": "0.4",
            "NEXUS_HCL_COST_WEIGHT": "0.3",
            "NEXUS_HCL_LATENCY_WEIGHT": "0.3",
        },
    ),
    capability(
        "image-gen-prefer",
        {"requires_image_gen": True, "image_gen_models": ["gemini-3.8-flash"]},
    ),
    Scenario(
        "latency-slope-shed",
        with_signal=Probe(delays_ms={"anthropic": (20.0, 70.0, 200.0)}, warmup=3),
        env={"NEXUS_LATENCY_SLOPE_WINDOW": "10", "NEXUS_LATENCY_SLOPE_THRESHOLD_MS": "25.0"},
    ),
    capability("logprobs-prefer", {"requires_logprobs": True}),
    capability(
        "long-context-prefer",
        {"min_context_tokens": 500000, "model_context_windows": {"gemini-4-argon": 1048576}},
    ),
    capability("mcp-prefer", {"requires_mcp": True, "mcp_models": ["gpt-6-astra"]}),
    capability(
        "moderation-prefer",
        {"requires_moderation": True, "moderation_models": ["gpt-6-luna"]},
    ),
    capability(
        "multimodal-input-prefer",
        {
            "has_images": True,
            "multimodal_models": ["gemini-3.8-flash"],
            "model_capabilities": {"gemini-3.8-flash": "vision"},
        },
    ),
    capability("pdf-input-prefer", {"requires_pdf": True, "pdf_models": ["gemini-4-argon"]}),
    capability(
        "predicted-outputs-prefer",
        {"requires_predicted_outputs": True, "predicted_outputs_models": ["gpt-6-luna"]},
    ),
    capability(
        "prompt-caching-prefer",
        {"requires_prompt_caching": True, "prompt_caching_models": ["claude-haiku-5-5"]},
    ),
    Scenario(
        "prompt-injection-risk-shed",
        with_signal=Probe(metadata={"prompt_injection_risk": 0.85}),
        env={"NEXUS_PROMPT_INJECTION_RISK_THRESHOLD": "0.7"},
    ),
    Scenario(
        "provider-circuit-probe",
        with_signal=Probe(failing=("anthropic",), warmup=3),
        env={"NEXUS_PROVIDER_CIRCUIT_PROBE_BUDGET": "1"},
    ),
    capability(
        "provider-exclusion",
        {"excluded_providers": "openai, moonshot", "excluded_models": ["claude-haiku-5-5"]},
    ),
    Scenario(
        "provider-hourly-cost-ceiling",
        with_signal=Probe(warmup=4),
        env={"NEXUS_PROVIDER_HOURLY_COST_CEILING_USD": "0.0001"},
        env_note="ceiling scaled down so four ~$0.00003 mock calls cross it",
    ),
    Scenario(
        "provider-spend-telemetry",
        with_signal=Probe(
            metadata={
                "spend:anthropic": "12.40",
                "spend:openai": "7.80",
                "spend:google": "1.90",
                "spend:moonshot": "3.30",
            }
        ),
        env={"NEXUS_PROVIDER_SPEND_SOFT_USD": "10.0"},
    ),
    Scenario(
        "provider-token-fair-share",
        with_signal=Probe(warmup=1),
        env={"NEXUS_PROVIDER_TOKEN_FAIR_SHARE_CEILING": "100000"},
    ),
    Scenario(
        "provider-warmup-weight",
        with_signal=Probe(metadata={"provider_warmup": {"openai": 0.9, "anthropic": 0.4}}),
        env={"NEXUS_PROVIDER_WARMUP_BLEND": "0.3"},
    ),
    Scenario(
        "provider-weight-decay",
        with_signal=Probe(failing=("anthropic",), warmup=1),
        env={"NEXUS_PROVIDER_WEIGHT_DECAY_FACTOR": "0.5", "NEXUS_PROVIDER_WEIGHT_RECOVER": "0.1"},
    ),
    Scenario("quality-weighted-sticky", with_signal=Probe(user="session-42")),
    Scenario(
        "queue-depth-fairness",
        with_signal=Probe(held_in_flight=("anthropic", 4)),
        env={"NEXUS_QUEUE_DEPTH_SOFT_CAP": "4"},
    ),
    capability(
        "reasoning-effort-prefer",
        {"requires_reasoning_effort": True, "reasoning_effort_models": ["gpt-6-astra"]},
    ),
    capability(
        "realtime-api-prefer",
        {"requires_realtime": True, "realtime_api_models": ["gemini-3.8-flash"]},
    ),
    Scenario(
        "region-carbon-blend",
        with_signal=Probe(
            metadata={
                "region": "eu",
                "carbon_intensity:anthropic": "200",
                "carbon_intensity:openai": "380",
            }
        ),
        env={"NEXUS_REGION_CARBON_BLEND_WEIGHT": "0.5"},
    ),
    Scenario(
        "region-failover-hysteresis",
        with_signal=Probe(failing=("anthropic", "google"), warmup=6),
        env={
            "NEXUS_REGION_FAILOVER_HYSTERESIS_SUCCESSES": "3",
            "NEXUS_STICKY_REGION_FAILOVER_PREFERENCES": '["eu","us","cn","global"]',
        },
    ),
    Scenario(
        "region-latency-p99-shed",
        with_signal=Probe(region="eu", delays_ms={"anthropic": (400.0,)}, warmup=2),
        env={"NEXUS_REGION_LATENCY_P99_MS": "150"},
        env_note="threshold scaled to the 400 ms mock provider",
    ),
    Scenario(
        "retry-after-respect",
        with_signal=Probe(retry_after={"anthropic": 30.0}, warmup=1),
        env={"NEXUS_RETRY_AFTER_DEFAULT_SECONDS": "30"},
    ),
    Scenario(
        "semantic-cache-ttl-affinity",
        with_signal=Probe(metadata={"cacheable": "true", "cache_ttl_remaining:google": "240"}),
        env={"NEXUS_SEMANTIC_CACHE_TTL_SECONDS": "300"},
    ),
    Scenario(
        "sticky-canary-cost",
        with_signal=Probe(metadata={"tenant_id": "acme"}),
        env={"NEXUS_STICKY_CANARY_COST_PERCENT": "10"},
    ),
    capability(
        "streaming-prefer",
        {
            "stream": True,
            "streaming_models": ["gemini-3.8-flash"],
            "model_capabilities": {"gemini-3.8-flash": "streaming"},
        },
    ),
    capability(
        "structured-output-prefer",
        {
            "requires_json": True,
            "structured_models": ["kimi-k3.1-pro"],
            "model_capabilities": {"kimi-k3.1-pro": "json"},
        },
    ),
    Scenario(
        "tenant-soft-isolation",
        with_signal=Probe(metadata={"tenant_id": "acme-corp", "tenant_rpm": 75}),
        env={"NEXUS_TENANT_SOFT_ISOLATION_RPM": "60"},
    ),
    Scenario(
        "thinking-model-prefer",
        with_signal=Probe(
            metadata={
                "complexity_score": 0.85,
                "thinking_models": ["gpt-6-astra", "gemini-4-argon"],
            }
        ),
        env={"NEXUS_THINKING_COMPLEXITY_THRESHOLD": "0.7"},
    ),
    Scenario(
        "token-bucket-burst",
        with_signal=Probe(warmup=10),
        env={"NEXUS_TOKEN_BUCKET_CAPACITY": "10", "NEXUS_TOKEN_BUCKET_REFILL_PER_SEC": "1.0"},
    ),
    Scenario(
        "token-rpm-ceiling",
        with_signal=Probe(warmup=2),
        env={"NEXUS_TOKEN_RPM_CEILING": "30"},
        env_note="ceiling scaled to the 13-token demo prompt",
    ),
    capability(
        "tool-calling-prefer",
        {
            "requires_tools": True,
            "tool_capable_models": ["kimi-k3.1-pro"],
            "model_capabilities": {"kimi-k3.1-pro": "tools"},
        },
    ),
    capability(
        "transcription-prefer",
        {"requires_transcription": True, "transcription_models": ["gemini-3.8-flash"]},
    ),
    capability("video-input-prefer", {"requires_video": True, "video_models": ["gemini-4-argon"]}),
    capability(
        "web-search-prefer",
        {
            "requires_web_search": True,
            "web_search_models": ["gemini-3.8-flash"],
            "model_capabilities": {"gemini-3.8-flash": "web_search"},
        },
    ),
)


@contextmanager
def environment(variables: Mapping[str, str]) -> Iterator[None]:
    """Expose ``NEXUS_*`` variables exactly as an operator would export them."""
    previous = {key: os.environ.get(key) for key in variables}
    os.environ.update(variables)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def quiet_stdout() -> Iterator[None]:
    """Silence the app's JSON logs and console trace spans at the descriptor level."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        os.dup2(devnull.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)


class Harness:
    """A fresh router, app wiring, and mock providers for one probe."""

    def __init__(self, scenario: Scenario, probe: Probe, label: str) -> None:
        """Build isolated router state for one probe.

        Args:
            scenario: Demo being rendered.
            probe: Probe whose setup (adapters, env) to apply.
            label: Unique state-file label for this probe.
        """
        self.scenario = scenario
        self.probe = probe
        self.hold = asyncio.Event() if probe.held_in_flight else None
        held_provider = probe.held_in_flight[0] if probe.held_in_flight else None
        self.adapters = {
            name: DemoAdapter(
                name,
                delays_ms=probe.delays_ms.get(name, ()),
                failing=name in probe.failing,
                retry_after=probe.retry_after.get(name),
                hold=self.hold if name == held_provider else None,
            )
            for name in PROVIDERS
        }
        with environment(scenario.env):
            settings = RouterSettings(
                _env_file=None,
                audit_log_path=str(STATE_DIR / f"{label}-audit.jsonl"),
                spend_ledger_path=str(STATE_DIR / f"{label}-spend.sqlite3"),
                idempotency_path=str(STATE_DIR / f"{label}-idempotency.sqlite3"),
                response_cache_enabled=False,
            )
        registry = AdapterRegistry(dict(self.adapters))
        self.router = NexusRouter(settings, registry)
        request_ids = count(1)
        main.get_settings = lambda: settings
        main.get_adapter_registry = lambda: registry
        main.get_router = lambda: self.router
        main.get_response_cache = lambda: ResponseCache(ttl_seconds=0.0, enabled=False)
        main.uuid4 = lambda: uuid.UUID(int=next(request_ids))
        self._next_id = request_ids

    def body(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return the chat-completions body sent for this probe."""
        payload: dict[str, Any] = {"messages": [{"role": "user", "content": PROMPT_TEXT}]}
        if self.probe.user:
            payload["user"] = self.probe.user
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        if self.probe.region is not None:
            payload["region"] = self.probe.region
        return payload

    async def post(
        self, client: httpx.AsyncClient, strategy: str, body: Mapping[str, Any]
    ) -> httpx.Response:
        """Send one chat completion through the in-process app."""
        return await client.post(
            "/v1/chat/completions", json=dict(body), headers={"X-Router-Strategy": strategy}
        )

    async def warm_up(self, client: httpx.AsyncClient) -> tuple[str, ...]:
        """Send real requests that build the probe's runtime state."""
        strategy = self.probe.warmup_strategy or self.scenario.strategy
        served: list[str] = []
        for _ in range(self.probe.warmup):
            response = await self.post(client, strategy, self.body())
            served.append(response.json()["model"] if response.status_code == 200 else "error")
        if self.probe.warmup_until_each_provider:
            target = self.probe.warmup_until_each_provider
            for _ in range(60):
                if all(adapter.calls >= target for adapter in self.adapters.values()):
                    break
                response = await self.post(client, strategy, self.body())
                served.append(response.json()["model"])
            else:
                raise RuntimeError(
                    f"{self.scenario.strategy}: warm-up never reached every provider"
                )
        return tuple(served)

    async def hold_in_flight(self, client: httpx.AsyncClient) -> list[asyncio.Task[httpx.Response]]:
        """Start requests that stay in flight on the held provider during the probe."""
        if self.probe.held_in_flight is None:
            return []
        provider, depth = self.probe.held_in_flight
        adapter = self.adapters[provider]
        tasks: list[asyncio.Task[httpx.Response]] = []
        for index in range(depth):
            tasks.append(
                asyncio.create_task(self.post(client, self.scenario.strategy, self.body()))
            )
            for _ in range(400):
                if adapter.calls > index:
                    break
                await asyncio.sleep(0.005)
            else:
                raise RuntimeError(
                    f"{self.scenario.strategy}: request {index} never reached {provider}"
                )
        return tasks

    async def run(self, *, http_metadata: Mapping[str, Any] | None = None) -> Outcome:
        """Run warm-up traffic, then the probe, and return the router's answer.

        Args:
            http_metadata: Send the probe over HTTP with this body ``metadata``
                (used to re-check that HTTP does not forward it).

        Returns:
            Status, served model, strategy, rationale, and warm-up models.
        """
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://nexus.local") as client:
            served = await self.warm_up(client)
            held = await self.hold_in_flight(client)
            if self.probe.in_process and http_metadata is None:
                outcome = await self._route_in_process()
            else:
                response = await self.post(client, self.scenario.strategy, self.body(http_metadata))
                outcome = self._outcome_from_http(response)
            if self.hold is not None:
                self.hold.set()
                await asyncio.gather(*held)
        return replace(outcome, warmup_served=served)

    def _outcome_from_http(self, response: httpx.Response) -> Outcome:
        """Read the routing decision from a chat-completions response.

        Raises:
            RuntimeError: If the app did not answer 200.
        """
        if response.status_code != 200:
            raise RuntimeError(
                f"{self.scenario.strategy}: HTTP {response.status_code}: {response.text}"
            )
        router = response.json()["router"]
        return Outcome(
            status=response.status_code,
            model=router["model_used"],
            provider=self.router.model_catalog[router["model_used"]].provider,
            strategy=router["routing_strategy"],
            rationale=router["rationale"],
        )

    async def _route_in_process(self) -> Outcome:
        """Route the probe through the same router with ``RouterRequest`` fields set."""
        request = RouterRequest(
            request_id=uuid.UUID(int=next(self._next_id)).hex,
            messages=[ChatMessage(content=PROMPT_TEXT)],
            user_id=self.probe.user or "anonymous",
            session_id=self.probe.user or "default",
            strategy=RoutingStrategyName(self.scenario.strategy),
            region=self.probe.region,
            metadata=dict(self.probe.metadata or {}),
        )
        response = await self.router.complete(request)
        return Outcome(
            status=200,
            model=response.model_used,
            provider=self.router.model_catalog[response.model_used].provider,
            strategy=response.routing_strategy.value,
            rationale=response.rationale,
        )


async def run_probe(
    scenario: Scenario, probe: Probe, label: str, *, http_metadata: Mapping[str, Any] | None = None
) -> Outcome:
    """Run one probe against a fresh router with app output silenced."""
    with quiet_stdout():
        return await Harness(scenario, probe, label).run(http_metadata=http_metadata)


def comparable(rationale: str, *, wall_clock: bool) -> str:
    """Return a rationale with wall-clock numbers masked when they can vary."""
    return re.sub(r"\d+(\.\d+)?", "#", rationale) if wall_clock else rationale


async def run_scenario(scenario: Scenario) -> tuple[Outcome, Outcome]:
    """Run both probes of a demo and re-check the HTTP metadata gap.

    Raises:
        RuntimeError: If the documented signal changes nothing, or HTTP now
            forwards the metadata/region the demo routes in-process.
    """
    without = await run_probe(scenario, scenario.without_signal, f"{scenario.strategy}-without")
    with_signal = await run_probe(scenario, scenario.with_signal, f"{scenario.strategy}-with")
    probe = scenario.with_signal
    if probe.in_process:
        wall_clock = bool(probe.delays_ms)
        over_http = await run_probe(
            scenario, probe, f"{scenario.strategy}-http", http_metadata=probe.metadata or {}
        )
        plain = replace(probe, metadata=None, region=None)
        baseline = await run_probe(scenario, plain, f"{scenario.strategy}-http-plain")
        if comparable(over_http.rationale, wall_clock=wall_clock) != comparable(
            baseline.rationale, wall_clock=wall_clock
        ):
            raise RuntimeError(
                f"{scenario.strategy}: HTTP now forwards metadata/region "
                f"({over_http.rationale!r} != {baseline.rationale!r}); "
                "route the demo over HTTP and drop the HTTP note from the guide"
            )
    if (without.model, without.rationale) == (with_signal.model, with_signal.rationale):
        raise RuntimeError(f"{scenario.strategy}: the documented signal changed nothing")
    return without, with_signal


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return Pillow's bundled default font at ``size`` (same on every machine)."""
    return ImageFont.load_default(size=size)


def wrap(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, width: int) -> list[str]:
    """Greedy word wrap to a pixel width, keeping indentation and hard-splitting long tokens."""
    stripped = text.lstrip(" ")
    indent = text[: len(text) - len(stripped)]
    width -= int(font.getlength(indent))
    lines: list[str] = []
    current = ""
    for word in stripped.split(" "):
        candidate = f"{current} {word}" if current else word
        if font.getlength(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        while font.getlength(word) > width:
            cut = len(word)
            while cut > 1 and font.getlength(word[:cut]) > width:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    if current:
        lines.append(current)
    return [f"{indent}{line}" for line in lines]


def compact_json(value: object) -> str:
    """Render JSON the way the guides show it, on one line."""
    return json.dumps(value, separators=(", ", ": "), ensure_ascii=False)


def request_lines(scenario: Scenario, probe: Probe, served: Sequence[str]) -> list[Line]:
    """Return the abridged request (and setup) shown for one probe."""
    lines: list[Line] = []
    setup = setup_summary(probe, served)
    if setup:
        lines.append((f"setup: {setup}", WARN))
    if probe.in_process:
        lines.append(("await router.complete(RouterRequest(", TEXT))
        lines.append((f'    strategy="{scenario.strategy}", messages=[same prompt],', TEXT))
        if probe.region is not None:
            lines.append((f'    region="{probe.region}",', ACCENT))
        if probe.metadata is not None:
            lines.append((f"    metadata={compact_json(dict(probe.metadata))}))", ACCENT))
        else:
            lines.append(("))", TEXT))
        return lines
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "Summarize this call..."}]}
    if probe.user:
        body["user"] = probe.user
    lines.append(("POST /v1/chat/completions", TEXT))
    lines.append((f"X-Router-Strategy: {scenario.strategy}", ACCENT))
    lines.append((compact_json(body), MUTED))
    return lines


def setup_summary(probe: Probe, served: Sequence[str]) -> str:
    """Describe the real traffic and mock-provider behaviour behind a probe."""
    parts: list[str] = []
    for provider in probe.failing:
        parts.append(f"mock {provider} answers HTTP 503")
    for provider, seconds in probe.retry_after.items():
        parts.append(f"mock {provider} answers HTTP 429 Retry-After {seconds:g}s")
    if probe.delays_ms:
        latency = ", ".join(
            f"{provider} {' then '.join(f'{value:g}' for value in values)}"
            for provider, values in probe.delays_ms.items()
        )
        parts.append(f"mock latency ms: {latency}")
    if probe.held_in_flight:
        provider, depth = probe.held_in_flight
        parts.append(f"{depth} requests held in flight on {provider}")
    if served:
        counts = Counter(served)
        strategy = probe.warmup_strategy or "same strategy"
        parts.append(
            f"{len(served)} warm-up requests ({strategy}) served by "
            + ", ".join(f"{model} x{number}" for model, number in counts.items())
        )
    return "; ".join(parts)


def decision_lines(outcome: Outcome) -> list[Line]:
    """Return the router's real decision for one probe."""
    status = f"{outcome.status} OK · " if outcome.status == 200 else f"{outcome.status} · "
    return [
        (f"-> {status}served {outcome.model} ({outcome.provider})", SUCCESS),
        (f"routing_strategy: {outcome.strategy}", MUTED),
        (f"rationale: {outcome.rationale}", TEXT),
    ]


@dataclass(frozen=True)
class Panel:
    """Text blocks rendered in one half of the frame."""

    heading: str
    request: list[Line]
    decision: list[Line]


def draw_block(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[Line],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    x: int,
    y: int,
    width: int,
) -> int:
    """Draw wrapped lines and return the next free y position."""
    for text, color in lines:
        for chunk in wrap(text, font, width):
            draw.text((x, y), chunk, fill=color, font=font)
            y += LINE_HEIGHT
    return y + 8


def render_frame(
    scenario: Scenario, panels: Sequence[Panel], stage: int, footer: str
) -> Image.Image:
    """Render one animation frame; ``stage`` reveals request/decision blocks in order.

    Raises:
        RuntimeError: If a panel's text would overflow the frame.
    """
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font, body_font = load_font(16), load_font(14)
    draw.rounded_rectangle(
        (12, 12, WIDTH - 12, HEIGHT - 12), radius=16, outline=(30, 41, 59), width=2
    )
    command = f"$ PYTHONPATH=src python scripts/render_strategy_demos.py --only {scenario.strategy}"
    draw.text((PADDING, PADDING), command, fill=PROMPT, font=title_font)
    guide = scenario.guide_path.relative_to(REPO_ROOT).as_posix()
    subtitle = f"{guide} · real decisions from the in-process app + mock providers (no network)"
    draw.text((PADDING, PADDING + 26), subtitle, fill=MUTED, font=body_font)
    if scenario.env:
        env = "  ".join(f"{key}={value}" for key, value in scenario.env.items())
        note = f"  ({scenario.env_note})" if scenario.env_note else ""
        draw.text((PADDING, PADDING + 48), f"env: {env}{note}", fill=ACCENT, font=body_font)
    column_width = (WIDTH - 2 * PADDING - 24) // 2
    top, bottom = PADDING + 84, HEIGHT - 70
    for index, panel in enumerate(panels):
        x = PADDING + index * (column_width + 24)
        draw.rounded_rectangle(
            (x - 8, top - 10, x + column_width + 8, bottom),
            radius=10,
            fill=(15, 23, 42),
            outline=(30, 41, 59),
        )
        if stage < index * 2 + 1:
            continue
        y = draw_block(draw, [(panel.heading, PROMPT)], title_font, x + 4, top, column_width - 8)
        y = draw_block(draw, panel.request, body_font, x + 4, y, column_width - 8)
        if stage >= index * 2 + 2:
            y = draw_block(draw, panel.decision, body_font, x + 4, y + 4, column_width - 8)
        if y > bottom:
            raise RuntimeError(f"{scenario.strategy}: panel {index + 1} overflows the frame")
    if stage >= 4:
        for offset, chunk in enumerate(wrap(footer, body_font, WIDTH - 2 * PADDING)):
            draw.text((PADDING, HEIGHT - 56 + offset * 18), chunk, fill=MUTED, font=body_font)
    return image


def shared_palette(frames: Sequence[Image.Image]) -> list[Image.Image]:
    """Map every frame onto one small palette so the GIF stays compact."""
    palette = frames[-1].quantize(
        colors=PALETTE_COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    return [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]


def with_signal_heading(probe: Probe) -> tuple[str, str]:
    """Return the right-hand panel heading and the frame footer for a probe."""
    if probe.in_process:
        return "2) with the documented signal (in-process)", HTTP_GAP_NOTE
    if probe.held_in_flight:
        return "2) with the signal (requests held in flight)", STATE_NOTE
    if probe.warmup or probe.warmup_until_each_provider:
        return "2) with the signal (after real warm-up traffic)", STATE_NOTE
    return "2) with the signal (over HTTP)", HTTP_NOTE


def render(scenario: Scenario, without: Outcome, with_signal: Outcome) -> list[Image.Image]:
    """Return GIF frames revealing both probes and the router's decisions."""
    heading, footer = with_signal_heading(scenario.with_signal)
    panels = (
        Panel(
            "1) without the signal (over HTTP)",
            request_lines(scenario, scenario.without_signal, without.warmup_served),
            decision_lines(without),
        ),
        Panel(
            heading,
            request_lines(scenario, scenario.with_signal, with_signal.warmup_served),
            decision_lines(with_signal),
        ),
    )
    stages = shared_palette(
        [render_frame(scenario, panels, stage, footer) for stage in range(1, 5)]
    )
    return [stages[0], stages[1], stages[1], stages[2], stages[3], stages[3], stages[3], stages[3]]


def main_cli(argv: Sequence[str] | None = None) -> int:
    """Run the demos and write (or just print) the GIFs."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", action="append", default=[], help="strategy to render (repeatable)"
    )
    parser.add_argument("--check", action="store_true", help="print decisions without writing GIFs")
    args = parser.parse_args(argv)
    known = {scenario.strategy for scenario in SCENARIOS}
    unknown = sorted(set(args.only) - known)
    if unknown:
        parser.error(f"no demo for: {', '.join(unknown)}")
    for key in [key for key in os.environ if key.startswith("NEXUS_")]:
        del os.environ[key]
    shutil.rmtree(STATE_DIR, ignore_errors=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    selected = [
        scenario for scenario in SCENARIOS if not args.only or scenario.strategy in args.only
    ]
    for scenario in selected:
        gif_path = scenario.gif_path
        without, with_signal = asyncio.run(run_scenario(scenario))
        print(f"{scenario.strategy}")
        print(f"  without: {without.model:18} {without.rationale}")
        print(f"  with:    {with_signal.model:18} {with_signal.rationale}")
        if with_signal.warmup_served:
            print(f"  warm-up: {', '.join(with_signal.warmup_served)}")
        if not args.check:
            save_gif(gif_path, render(scenario, without, with_signal), FRAME_MS)
    shutil.rmtree(STATE_DIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
