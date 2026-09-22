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

Spend is tracked by user key. Nexus rejects requests when estimated next spend would exceed `NEXUS_BUDGET_CAP_USD`.

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

The API key identifier is derived from the `Authorization` header.

## Failure Semantics

Provider failures do not immediately fail the request. Nexus records the failure, updates provider metrics, opens circuits when needed, and attempts the configured fallback chain. The request fails only when every eligible attempt fails.

### Guardrail rejections are not provider failures

Client-side guardrails are evaluated per candidate *before* dispatch:

- **Budget cap** (`BudgetExceededError`)
- **Open circuit** (`CircuitOpenError`)

When a guardrail rejects a candidate, Nexus **skips** that candidate and moves to
the next entry in the fallback chain **without** recording a provider failure or
incrementing `provider_error_rate`. This keeps a provider's circuit breaker a
signal of provider health only: a user hitting their spend cap, or a circuit that
is already open, can never accumulate spurious failures that would open (or keep
open) a circuit for otherwise-healthy traffic. Only genuine dispatch errors
(timeouts, adapter/HTTP errors) count toward the circuit breaker.

## Prompt injection gateway

`PromptInjectionGateway` can allow, redact, or block high-risk prompt patterns before routing. Advisory control complementary to routing shed strategies.

## Request fingerprint deduper

`RequestFingerprintDeduper` hashes `model + messages + temperature + tenant` into a
SHA-256 fingerprint and flags identical payloads inside a short TTL window. It is
distinct from `IdempotencyStore` (durable client-key response replay): this control
never stores response bodies and never rejects traffic — callers decide how to
handle `is_duplicate` for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 bursts.

## First-token latency SLO advisor

`FirstTokenLatencySloAdvisor` classifies time-to-first-token (TTFT) into `within` /
`warn` / `breach` bands against a default or per-model `slo_ttft_ms`. It is distinct
from `ModelLatencySlaTracker` (end-to-end p50/p95 windows): this control is for
streaming first-token latency only and never rejects traffic for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Provider fallback chain planner

`ProviderFallbackChainPlanner` builds deterministic ordered provider fallback
chains from explicit preference lists, skipping unavailable providers and
optionally anchoring on a caller-supplied primary. It is distinct from
`ProviderFallbackScoreboard` (live health ranking via `record_outcome` / `rank()`):
this planner never observes latency or errors — preference order only — for
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Prompt cache hit-rate advisor

`PromptCacheHitRateAdvisor` classifies `cached_tokens / prompt_tokens` into
`cold` / `warm` / `hot` bands. It is distinct from `RequestFingerprintDeduper`
(identical-request content-hash windows) and `SemanticCacheStrategy`
(near-duplicate response routing): this control never rejects traffic and never
stores response bodies — callers log or alert on `band` for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Output token ceiling guard

`OutputTokenCeilingGuard` classifies `requested_max_tokens` against a policy
ceiling into `ok` / `near` / `over` bands. It is distinct from
`StreamingTokenBudgetGate` (hard mid-stream cut-off) and
`ContextWindowFitAdvisor` (prompt tokens vs model context window): this control
never kills requests itself — callers clamp, warn, or proceed for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Provider region affinity advisor

`ProviderRegionAffinityAdvisor` ranks providers by affinity to a
`preferred_regions` list using a provider→regions map. It is distinct from
`ProviderHealthScoreboard` (success/error health bands) and
`StickyProviderAffinity` (session sticky map): this control never rejects
traffic and never mutates sticky state — callers reorder or log from
`rankings` for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Model temperature clamp advisor

`ModelTemperatureClampAdvisor` classifies requested `temperature` against a
policy `max_allowed` into `ok` / `high` / `extreme` bands and returns a
`suggested_clamp`. It never rejects traffic itself — callers clamp, warn, or
proceed for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.


### StructuredOutputRepairAdvisor
Advisory accept/repair/fail bands for structured JSON outputs against required keys. Never auto-forwards invalid payloads. Prefer GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 for narrative repair prompts. See `docs/guides/STRUCTURED_OUTPUT_REPAIR_ADVISOR_GUIDE.md`.

### ProviderWarmPoolAdvisor
Advisory cold/warming/hot bands from warm replica counts vs target. Never auto-routes. Prefer GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. See `docs/guides/PROVIDER_WARM_POOL_ADVISOR_GUIDE.md`.

### CostAttributionTagLedger
Offline Helicone-style cost attribution tags with per-tag rollups. Never enforces spend caps. Prefer GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. See `docs/guides/COST_ATTRIBUTION_TAG_LEDGER_GUIDE.md`.
