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

With tool calling, the scrubber also covers the tool-call arguments of replayed
assistant turns and `tool` results. They are scrubbed as JSON, leaf by leaf
(every string, object key, and number; a number that would be redacted becomes
the redaction string), so they stay valid JSON; plain-text tool results are
scrubbed as text. Tool-call ids and function names are kept, so calls still pair
with their results. As for text answers, the model's new calls are returned
unscrubbed.

## Rate Limiting

Nexus uses a token bucket per API key identifier:

- Capacity: `NEXUS_RATE_LIMIT_CAPACITY`
- Refill rate: `NEXUS_RATE_LIMIT_REFILL_PER_SECOND`

The API key identifier is a non-reversible fingerprint of the `Authorization`
header (`key_` + 16 hex chars of SHA-256 over the credential, with any `Bearer`
prefix stripped; `anonymous` when absent). The raw header is never used as an
identifier, so it cannot leak through 429 details, `GET /v1/spend`, or the
durable spend/idempotency stores. In `virtual_keys` mode the identifier is the
issued key's id instead, and each key's own `rpm_limit` applies on top (see
[Virtual keys](#virtual-keys)). See [SECURITY.md](SECURITY.md#credential-handling).

## Virtual keys

With `NEXUS_AUTH_MODE=virtual_keys` only the master key and keys it issued
reach a model endpoint (see
[docs/guides/VIRTUAL_KEYS_GUIDE.md](docs/guides/VIRTUAL_KEYS_GUIDE.md)):

- **Key handling.** Secrets are 256 random bits behind `sk-nexus-`, returned
  once (create / rotate) and stored only as HMAC-SHA-256 under
  `NEXUS_KEY_HASH_PEPPER`, with a 4-character display prefix. Stored digests
  are compared with `hmac.compare_digest`; a copy of the key store cannot
  verify a leaked key without the pepper.
- **Master key.** `NEXUS_MASTER_KEY` is a secret setting (masked in reprs and
  settings validation errors), compared in constant time over fixed-length
  digests, required at startup in `virtual_keys` mode, and never logged.
- **No secrets in logs or stores.** The spend ledger, audit log, response
  cache, idempotency and feedback stores, structured logs, and error messages
  carry the key id (`vk_...`) or display prefix only; unknown keys are never
  echoed back.
- **Refusals before spend.** Unknown, revoked, expired, and blocked keys get a
  401, a catalog model outside the allowlist a 403, an exhausted window budget
  a 402, and an exceeded RPM limit a 429, all before any provider call.
  Routing and fallbacks never leave the allowlist, and candidates whose cost
  forecast exceeds the key's remaining budget are skipped like the global
  budget guardrail's refusals (no provider failure is recorded).
- **Accounting failures fail closed.** When a virtual key's spend cannot be
  written to the ledger, the answer is returned, the row is retained, and the
  key's next metered request retries it, getting 503 `spend_unsettled` until
  it is recorded.
- **Limits of the guarantee.** Budgets are checked against settled spend, so
  requests in flight when a budget runs out can overshoot it by their own
  cost; RPM buckets are per process; `/health` and `/metrics` stay
  unauthenticated.

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

### Accounting write failures are not provider failures

Once a provider has answered, it has also billed the request. Recording the
request afterwards (spend ledger, audit log, feedback index) is the gateway's
own work, so a failure there:

- never sends the request to another candidate, and never records a
  circuit-breaker failure, success-rate or scoreboard sample, or
  `provider_error_rate` increment against the provider that answered;
- never fails the request when the write hits an I/O error (SQLite
  `database is locked`, a full or read-only disk). The caller gets the answer
  and the in-memory budget guardrail still counts its spend. Nexus logs
  `spend_ledger_write_failed` (request id, spend subject, provider, model,
  cost, and token counts, enough to restore the missing `/v1/spend` row),
  `audit_log_write_failed` (request id), or `feedback_index_failed`. Alert on
  these warnings: the durable ledger or audit log is missing those requests.

Any other error after the answer is a gateway bug: Nexus logs
`response_recording_failed` (request id, provider, error) and fails the request
once, without a fallback or a provider failure, with HTTP 500 `server_error`
(code `response_recording_failed`; `api_error` on `/v1/messages`) or, when a
stream has already started, the surface's error event (a chat error frame,
`response.failed`, or an Anthropic `error` event). Before this, a locked ledger
or a full audit volume sent the answered request to every fallback model (each
one billed again), counted a failure against every provider that had answered,
and returned HTTP 500; a fully streamed answer ended with an error that blamed
the provider.

## Request feedback

`POST /v1/feedback` lets callers rate completed requests, and the
`feedback-aware` strategy routes by those ratings. Because feedback changes
routing, it is guarded against poisoning:

- **Tenant scoping.** Requests, feedback, summaries, and routing statistics are
  partitioned by the API key fingerprint plus `X-Tenant-Id` recorded for each
  request. Feedback from any other key or tenant gets the same 404 as an unknown
  request id, so callers cannot probe for other tenants' requests, and one
  tenant's feedback never routes another tenant's traffic. The response cache
  is namespaced the same way, so cached answers (and their request ids) are
  never replayed across callers.
- **Request-bound, idempotent votes.** Feedback must reference a completed
  request of the caller inside `NEXUS_FEEDBACK_WINDOW_SECONDS`, and each
  `(request_id, metric, source)` keeps one value: retries replace a vote instead
  of adding one. A caller's influence is bounded by its own rate-limited,
  budgeted traffic, at most one human and one automated value per request and
  metric.
- **Minimum samples.** Below `NEXUS_FEEDBACK_MIN_SAMPLES` values across a
  request's candidates, `feedback-aware` is exactly `rule-based`.
- **Bounded influence.** `feedback-aware` only reorders the models `rule-based`
  would try, minus fallbacks outside the request's domain or latency class, open
  circuits, and models the engine would refuse before dispatch (no known price,
  or over the caller's budget); feedback can never route to anything else, and
  refusals keep the 402 `budget_exceeded` / 500 `model_price_unknown` statuses. Thompson sampling starts every model at `Beta(1, 1)`, and
  feedback older than the window stops counting.
- **Typed metrics.** A metric keeps one value kind (boolean or numeric) per
  caller; a conflicting write is a 409, so `1` cannot silently replace `true`.
- **Failure isolation.** A feedback-store error never fails a request: indexing
  errors are logged as `feedback_index_failed`, and `feedback-aware` falls back
  to `rule-based` when it cannot read feedback.

Privacy: comments are stored in `NEXUS_FEEDBACK_PATH` and returned only to the
caller that wrote them; they are never logged and never exported as metrics.
Feedback logs (`feedback_recorded`, `feedback_rejected`) and the
`router_feedback_total{metric, model}` counter carry no tenant ids or key
fingerprints, and client-chosen metric names beyond 32 per process share the
`other` label. See
[docs/guides/FEEDBACK_LOOP_GUIDE.md](docs/guides/FEEDBACK_LOOP_GUIDE.md#safety-privacy-and-limits).

## Tool calling

`POST /v1/chat/completions` routes OpenAI-style tool (function) calling. The
controls:

- **Nexus never executes tools.** There is no tool registry, code execution, or
  call to any tool endpoint: nexus forwards the model's `tool_calls` to the
  client and the client's `role: "tool"` results to the model. The application
  decides whether to run each call.
- **Tool calls are untrusted model output.** A prompt-injected document or tool
  result can make a model request a harmful call. Validate arguments against
  your own schema, allowlist side-effecting functions, and have a human confirm
  destructive actions; nexus only checks that replayed arguments are a JSON
  object, not that they match the schema.
- **Tool results are forwarded as-is** (after optional PII scrubbing) and are
  not scanned for prompt injection; keep secrets out of them.
- **Validated before routing.** Tool schemas (at most 128 function tools,
  `^[a-zA-Z0-9_-]{1,64}$` names, object parameter schemas), `tool_choice`, and
  the pairing of calls and results are checked first, so malformed conversations
  get an OpenAI-style 400 and never reach a provider.
- **Capability gate.** Requests that offer or replay tools only go to models
  whose capability data lists `tools`; when the routing decision has none, the
  request fails with 400 `tool_calling_unsupported` before any provider call
  (no spend). The gate never adds a model outside the strategy's decision.
- **Names and counts only.** The audit log records `tool_count` and
  `tool_call_names`, the `routing_response` log adds `tools_offered` and the
  called names, and tool-calling error messages name indices, ids, and function
  names; tool arguments and results never reach logs or the audit trail.
- **Isolation.** Response-cache keys include the tools, `tool_choice`, replayed
  calls, and `tool_call_id`s, so an answer is never replayed for another tool set
  or another tool result, and cached or idempotent replays stay in the caller's
  namespace.

See [docs/guides/TOOL_CALLING_GUIDE.md](docs/guides/TOOL_CALLING_GUIDE.md#safety).

## Prompt injection gateway

`PromptInjectionGateway` can allow, redact, or block high-risk prompt patterns before routing. Advisory control complementary to routing shed strategies.

## Model deprecation guard

`ModelDeprecationGuard` (`src/safety/model_deprecation.py`) maps retired model IDs
to current replacements. A deprecated model that still answers gets `warn`; from
its retirement date (UTC), when requests to it fail at the provider, it gets
`block`. The verdict's `reason` names the replacement and `retires_on` carries the
date. Lifecycle verified 2026-11-24: `claude-sonnet-4-5` blocks from 2026-11-30
and `claude-opus-4-5` from 2027-01-22 (with their dated IDs), and
`claude-3-5-sonnet`, retired on 2025-10-28, blocks now. It is a library check:
nexus does not call it on the request path. See the
[Model deprecation guard guide](docs/guides/MODEL_DEPRECATION_GUARD_GUIDE.md).

## Request fingerprint deduper

`RequestFingerprintDeduper` hashes `model + messages + temperature + tenant` into a
SHA-256 fingerprint and flags identical payloads inside a short TTL window. Pass
`tools=ToolConfig.canonical()` to include the offered tools, `tool_choice`, and
`parallel_tool_calls`; digests of requests without tools are unchanged. It is
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

