# Safety Controls

Nexus applies safety controls before and during provider dispatch.

## Circuit Breakers

Each provider has an independent circuit breaker:

- Opens after 3 consecutive failures.
- Stays open for 60 seconds.
- Recovers on the next allowed attempt after the recovery window.

## Request Timeouts

Provider calls are wrapped in `asyncio.wait_for` and adapter-level `httpx` timeouts. Default timeout is 30 seconds.

## Budget Guardrail

Spend is tracked by user key. Nexus rejects requests when estimated next spend would exceed `NEXUS_BUDGET_CAP_USD`: when the cap refuses every candidate model, the request fails with HTTP 402 `budget_exceeded` (`billing_error` on `/v1/messages`; a `budget_exceeded` error frame on chat-completion streams).
Recorded spend uses the provider adapter's price for the model, or the catalog price when the adapter has none; a model priced by neither is never dispatched (see [CONFIGURATION.md](CONFIGURATION.md#how-completions-are-priced)).

## PII Scrubbing

PII scrubbing is disabled by default. When enabled, Nexus runs regex redaction for:

- Email addresses
- US phone numbers

Presidio can be installed with:

```bash
python -m pip install ".[pii]"
```

Then inject Presidio analyzer and anonymizer engines into `PiiScrubber` during application composition.

## Rate Limiting

Nexus uses a token bucket per API key identifier:

- Capacity: `NEXUS_RATE_LIMIT_CAPACITY`
- Refill rate: `NEXUS_RATE_LIMIT_REFILL_PER_SECOND`

The API key identifier is a non-reversible fingerprint of the `Authorization`
header (`key_` + 16 hex chars of SHA-256 over the credential, with any `Bearer`
prefix stripped; `anonymous` when absent). The raw header is never used as an
identifier, so it cannot leak through 429 details, `GET /v1/spend`, or the
durable spend/idempotency stores. See [SECURITY.md](SECURITY.md#credential-handling).

## Failure Semantics

Provider failures do not immediately fail the request. Nexus records the failure, updates provider metrics, opens circuits when needed, and attempts the configured fallback chain. The request fails only when every eligible attempt fails.

### Guardrail rejections are not provider failures

Client-side guardrails are evaluated per candidate *before* dispatch:

- **Budget cap** (`BudgetExceededError`)
- **Unpriced model** (`UnpricedModelError`): neither its adapter nor the catalog
  prices it, so its spend could not be budgeted. When no candidate has a
  price, the request fails with HTTP 500 `model_price_unknown`, a gateway
  configuration error
- **Open circuit** (`CircuitOpenError`)

When a guardrail rejects a candidate, Nexus **skips** that candidate and moves to
the next entry in the fallback chain **without** recording a provider failure or
incrementing `provider_error_rate`. This keeps a provider's circuit breaker a
signal of provider health only: a user hitting their spend cap, or a circuit that
is already open, can never accumulate spurious failures that would open (or keep
open) a circuit for otherwise-healthy traffic. Only genuine dispatch errors
(timeouts, network errors, and the provider HTTP errors listed below) count
toward the circuit breaker.

### Provider rejections of the request are not provider failures

A provider that answers HTTP **400, 413, or 422** is rejecting the request
itself (malformed input, a prompt too large for the model), not failing.
Nexus returns that status to the caller (`provider_rejected_request` on
`/v1/responses`, `invalid_request_error` in streams) and:

- does **not** try the rest of the fallback chain, which would only repeat the
  rejection;
- does **not** record a circuit-breaker failure, success-rate or scoreboard
  sample, or `provider_error_rate` increment (the rejection is logged as
  `provider_request_rejected`).

Without this, a few malformed requests from any client opened the circuit of
every provider in the fallback chain and failed all tenants' traffic with 503
for the recovery window. Conversations that not every provider can answer (an
empty `messages` array, or only system instructions, which Anthropic and Gemini
reject) are refused up front with an OpenAI-style 400 on both
`/v1/chat/completions` and `/v1/responses`.

Every other provider error still counts toward the circuit breaker and falls
back: 5xx, timeouts, 408, 409, 429, and 401/403/404 (gateway credentials or a
model the provider does not serve are deployment faults, not client errors).
A provider that rejects *every* request with 400 (Anthropic, for example,
reports an exhausted credit balance that way) therefore no longer fails over
automatically; alert on `provider_request_rejected` log volume per provider.

## Prompt injection gateway

`PromptInjectionGateway` can allow, redact, or block high-risk prompt patterns before routing. Advisory control complementary to routing shed strategies.

## Request fingerprint deduper

`RequestFingerprintDeduper` hashes `model + messages + temperature + tenant` into a
SHA-256 fingerprint and flags identical payloads inside a short TTL window. It is
distinct from `IdempotencyStore` (durable client-key response replay): this control
never stores response bodies and never rejects traffic — callers decide how to
handle `is_duplicate` for GPT-6 / Claude Opus 5.5 / Gemini 4 / Kimi K3.1 bursts.

## First-token latency SLO advisor

`FirstTokenLatencySloAdvisor` classifies time-to-first-token (TTFT) into `within` /
`warn` / `breach` bands against a default or per-model `slo_ttft_ms`. It is distinct
from `ModelLatencySlaTracker` (end-to-end p50/p95 windows): this control is for
streaming first-token latency only and never rejects traffic for GPT-6 /
Claude Opus 5.5 / Gemini 4 / Kimi K3.1.

## Provider fallback chain planner

`ProviderFallbackChainPlanner` builds deterministic ordered provider fallback
chains from explicit preference lists, skipping unavailable providers and
optionally anchoring on a caller-supplied primary. It is distinct from
`ProviderFallbackScoreboard` (live health ranking via `record_outcome` / `rank()`):
this planner never observes latency or errors — preference order only — for
GPT-6 / Claude Opus 5.5 / Gemini 4 / Kimi K3.1.

## Prompt cache hit-rate advisor

`PromptCacheHitRateAdvisor` classifies `cached_tokens / prompt_tokens` into
`cold` / `warm` / `hot` bands. It is distinct from `RequestFingerprintDeduper`
(identical-request content-hash windows) and `SemanticCacheStrategy`
(near-duplicate response routing): this control never rejects traffic and never
stores response bodies — callers log or alert on `band` for GPT-6 /
Claude Opus 5.5 / Gemini 4 / Kimi K3.1.

## Output token ceiling guard

`OutputTokenCeilingGuard` classifies `requested_max_tokens` against a policy
ceiling into `ok` / `near` / `over` bands. It is distinct from
`StreamingTokenBudgetGate` (hard mid-stream cut-off) and
`ContextWindowFitAdvisor` (prompt tokens vs model context window): this control
never kills requests itself — callers clamp, warn, or proceed for GPT-6 /
Claude Opus 5.5 / Gemini 4 / Kimi K3.1.

## Provider region affinity advisor

`ProviderRegionAffinityAdvisor` ranks providers by affinity to a
`preferred_regions` list using a provider→regions map. It is distinct from
`ProviderHealthScoreboard` (success/error health bands) and
`StickyProviderAffinity` (session sticky map): this control never rejects
traffic and never mutates sticky state — callers reorder or log from
`rankings` for GPT-6 / Claude Opus 5.5 / Gemini 4 / Kimi K3.1.

## Model temperature clamp advisor

`ModelTemperatureClampAdvisor` classifies requested `temperature` against a
policy `max_allowed` into `ok` / `high` / `extreme` bands and returns a
`suggested_clamp`. It never rejects traffic itself — callers clamp, warn, or
proceed for GPT-6 / Claude Opus 5.5 / Gemini 4 / Kimi K3.1.


### StructuredOutputRepairAdvisor
Advisory accept/repair/fail bands for structured JSON outputs against required keys. Never auto-forwards invalid payloads. Prefer GPT-6 / Claude Opus 5.5 / Gemini 4 / Kimi K3.1 for narrative repair prompts. See `docs/guides/STRUCTURED_OUTPUT_REPAIR_ADVISOR_GUIDE.md`.

### ProviderWarmPoolAdvisor
Advisory cold/warming/hot bands from warm replica counts vs target. Never auto-routes. Prefer GPT-6 / Claude Opus 5.5 / Gemini 4 / Kimi K3.1. See `docs/guides/PROVIDER_WARM_POOL_ADVISOR_GUIDE.md`.

### CostAttributionTagLedger
Offline Helicone-style cost attribution tags with per-tag rollups. Never enforces spend caps. Prefer GPT-6 / Claude Opus 5.5 / Gemini 4 / Kimi K3.1. See `docs/guides/COST_ATTRIBUTION_TAG_LEDGER_GUIDE.md`.
### ToolLoopDepthGateAdvisor
`ToolLoopDepthGateAdvisor` advises `loop_depth` bands for HITL review. Never auto-blocks production traffic without human policy. Gap vs LiteLLM/vLLM/OpenAI-compatible tool-loop depth gates.

### WebGroundingInjectionGateAdvisor
`WebGroundingInjectionGateAdvisor` advises `injection_score` bands for HITL review. Never auto-blocks production traffic without human policy. Gap vs Perplexity/Bing/OpenAI web-grounding injection gates.

### EmbeddingCacheThrashAdvisor
`EmbeddingCacheThrashAdvisor` advises `thrash_ratio` bands for HITL review. Never auto-blocks production traffic without human policy. Gap vs Redis/vLLM/embedding-cache thrash monitors.

