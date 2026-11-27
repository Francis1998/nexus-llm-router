# Architecture

Nexus is an opinionated AI infrastructure component built around `Observe -> Decide -> Act`.

## Request Lifecycle

```text
RECEIVED
  -> CLASSIFIED
  -> ROUTED
  -> DISPATCHED
  -> RESPONDED
```

Fallback path:

```text
DISPATCHED -> FALLBACK -> DISPATCHED -> RESPONDED
```

Failure path:

```text
RECEIVED | CLASSIFIED | ROUTED | DISPATCHED | FALLBACK -> FAILED
```

## Observe

`src/router/analyzer.py` extracts:

- Complexity score from `LogisticComplexityClassifier`
- Domain tag from deterministic domain features
- Latency requirement from prompt size and complexity
- Token budget and estimated prompt tokens

## Decide

`src/router/strategies.py` contains pluggable strategies:

- `RuleBasedStrategy`
- `ClassifierStrategy`
- `CostOptimalStrategy`
- `LatencyAwareStrategy`
- `ABRoutingStrategy`
- `FeedbackAwareStrategy` (reorders `RuleBasedStrategy`'s candidates by request feedback; see [Feedback Loop](#feedback-loop))

Strategies return `RoutingDecision`, which includes selected model, provider, strategy, rationale, and fallback chain.

## Act

`src/router/engine.py` dispatches through `BaseProviderAdapter`:

- `complete()`
- `stream()`
- `estimate_cost()`
- `has_price()`
- `health_check()`

Concrete adapters live in `src/adapters/`. A completion's `cost_usd` comes from
the adapter's price table; for a catalog model the table lacks (`has_price()` is
`False`) the engine bills the catalog price instead, and a model priced nowhere
is never dispatched. Announced price changes (`PRICE_SCHEDULE` in
`src/router/config.py`) apply to both from their UTC `effective_from`;
`estimate_cost(..., at=...)` prices an instant, and the engine fixes each
request's prices once, when it routes the request.

### Response normalization contract

Each adapter normalizes a provider-specific payload into a `ProviderResponse`
(`content`, `model`, `input_tokens`, `output_tokens`, `cost_usd`, `finish_reason`). Because
provider content is a *list* (OpenAI `choices`, Gemini `content.parts`,
Anthropic `content` blocks), an adapter must reconstruct the full text rather
than reading only the first element: it concatenates every text segment in
order and skips non-text parts (for example Gemini `functionCall` parts, or
Anthropic `thinking`/`tool_use` blocks that can precede the answer). Reading a
single element silently truncates multi-part completions and returns an empty
string whenever the leading element is not itself a text segment.

The same rule applies to a single message's `content`: OpenAI-compatible
gateways may return `message.content` as a structured list of
`{"type": "text", "text": ...}` parts rather than a string. The shared
`adapters.http_utils.message_text` helper accepts either shape and joins the
text parts, so the OpenAI and Moonshot adapters never drop a valid completion.

Adapters also normalize the provider stop reason into
`ProviderResponse.finish_reason` with `adapters.http_utils.finish_reason`.
`length` means the output-token cap was hit (OpenAI/Kimi `length`, Anthropic
`max_tokens`, Gemini `MAX_TOKENS`); every other reason is `stop`. `stream()`
yields text chunks and then ends with the final `ProviderResponse`, so the
engine records provider-reported usage, cost, and stop reason for streams too.
A custom adapter whose stream yields text only falls back to estimates.

## API Surface

`src/api/main.py` exposes `/v1/chat/completions`, the OpenAI Responses API
(`/v1/responses`, translated by `src/api/responses.py`), and the Anthropic
Messages API (`/v1/messages` and `/v1/messages/count_tokens`, translated by
`src/api/anthropic_messages.py`) over the same router. All three share one
ingress path: the `Authorization` header (or, on the Anthropic paths, an
equivalent `x-api-key`) is reduced to a non-reversible key fingerprint,
`X-Tenant-Id` hard limits and per-endpoint `Idempotency-Key` replay run before
routing, and non-streaming requests use the exact-match response cache. Router
failures map to 429 (rate limit), 402 (the budget guardrail refused every
candidate), 500 (no candidate model has a price), and 503 (an affordable,
priced candidate failed at its provider); a provider's 400 / 413 / 422 rejection of the
request itself is returned with its status, without fallback or a
circuit-breaker failure. Errors are rendered as OpenAI envelopes on
`/v1/responses` and Anthropic envelopes on `/v1/messages*`. Responses and
Messages streams pull the first non-empty chunk before emitting their SSE
events so failures before any output stay real HTTP errors;
`NexusRouter.complete_stream`'s `on_complete` callback supplies the final usage
for `response.completed` and `message_delta`. The Messages translation is
text-only (tools, thinking, and non-text blocks are rejected with a 400) and
enforces `stop_sequences` at the gateway because adapters cannot pass them
upstream; `count_tokens` returns the analyzer's prompt estimate without calling
a provider.
`complete_stream` falls back to the next candidate only while no text has been
streamed; after that a provider failure ends the stream instead of splicing a
second answer onto the partial one. `GET /v1/models` and `GET /v1/models/{id}`
(`src/api/models.py`) list the routable catalog in the OpenAI format, with a
`nexus` block for pricing, token limits, capabilities, and live provider health
(adapter credentials and circuit-breaker state). `POST /v1/feedback` and
`GET /v1/feedback/summary` (`src/api/feedback.py`) rate completed requests and
aggregate the ratings; see [Feedback Loop](#feedback-loop). The response cache
is namespaced by API key fingerprint, `X-Tenant-Id`, and end user, so a cached
answer and its `request_id` are only replayed to the caller allowed to rate it.

## Request Admission

Before routing, `_admit_caller` in `src/api/main.py` turns the request's
credential into the caller identity every later stage keys on (`api_key_id`):

```text
Authorization / x-api-key
  -> NEXUS_AUTH_MODE=open:         fingerprint (key_<16 hex>), no limits
  -> NEXUS_AUTH_MODE=virtual_keys: KeyGate.admit (src/api/keys.py)
       1. credential: master key (constant-time) or VirtualKeyStore.authenticate
          (HMAC-SHA-256 lookup)                                        -> 401
       2. tenant binding vs X-Tenant-Id                                 -> 403
       3. explicit catalog model vs the key's allowlist                 -> 403
       4. metered endpoints: retry a failed ledger write of the key      -> 503
          then window spend from the router's spend ledger              -> 402
       5. metered endpoints: per-key RPM token bucket                   -> 429
       -> api_key_id = vk_..., KeyLimits(allowed_models, remaining_budget_usd)
  -> X-Tenant-Id hard rate limit -> Idempotency-Key -> response cache
  -> NexusRouter: decision and fallback chain restricted to allowed_models;
     candidates whose estimate exceeds remaining_budget_usd are skipped (402)
  -> provider -> spend ledger row (api_key_id, real cost) + audit record
```

`KeyLimits` (`src/router/schemas.py`) is the only thing the router learns about
virtual keys: `NexusRouter._restrict_to_key_models` filters the strategy's
decision and `_assert_key_budget` joins the per-candidate budget checks, so
every strategy and every fallback respects a key's policy without knowing
about keys. Spend is never cached per process: each admission reads the key's
settled spend in its UTC window from the shared ledger, so all workers enforce
the same budget. `/v1/keys*` (`KeyAdmin`) is master-key only; the store keeps
hashes and limits in `NEXUS_VIRTUAL_KEYS_PATH`. See
[docs/guides/VIRTUAL_KEYS_GUIDE.md](docs/guides/VIRTUAL_KEYS_GUIDE.md).

## Audit Log

`AuditLog` persists newline-delimited JSON to `NEXUS_AUDIT_LOG_PATH`. Each response records request id, chosen model, provider, strategy, rationale, latency, token cost, token usage, final state, the caller's API key fingerprint, or virtual key id in `virtual_keys` mode (`api_key_id`), and `X-Tenant-Id` (`tenant_id`, a tenant-bound key's tenant), and the analyzer's `domain`. The spend ledger records the same `api_key_id` per request.

The audit, spend-ledger, and feedback-index writes run after the provider answered, outside the attempt loop's failure handling: an I/O error there logs a warning (`audit_log_write_failed`, `spend_ledger_write_failed`, `feedback_index_failed`) and still returns the answer, and no error after the answer triggers a fallback or counts against the provider (see [SAFETY.md](SAFETY.md#accounting-write-failures-are-not-provider-failures)).

## Feedback Loop

Feedback sits after `RESPONDED` and feeds the next requests' `Decide` phase:

```text
RESPONDED -> audit log append -> FeedbackStore.register_request (request index)
POST /v1/feedback -> ownership check (key fingerprint + tenant) -> upsert + hourly totals
next request with X-Router-Strategy: feedback-aware -> Decide reads the totals
```

1. After a provider answers, `NexusRouter._respond` appends the audit record and
   registers the request in `FeedbackStore` (`src/router/feedback.py`,
   `NEXUS_FEEDBACK_PATH`, SQLite in WAL mode). Registration errors are logged as
   `feedback_index_failed` and never fail or retry the request.
2. `POST /v1/feedback` checks that the request id belongs to the caller's scope
   and is inside `NEXUS_FEEDBACK_WINDOW_SECONDS`, then upserts one value per
   `(request_id, metric, source)` with the model, provider, strategy, and domain
   copied from the request index, and updates hourly per-model totals in the same
   transaction.
3. `FeedbackAwareStrategy` reads those totals for the caller, metric, and domain,
   and Thompson-samples among `RuleBasedStrategy`'s attempts (filtered by domain,
   latency class, open circuits, and `NexusRouter.allows_dispatch`, the
   read-only twin of the price and budget checks) with a generator seeded by
   the request id. Below `NEXUS_FEEDBACK_MIN_SAMPLES` it returns the
   `rule-based` decision (cold start).

## Observability

Nexus emits:

- Structured JSON logs through `structlog`
- Prometheus metrics from `/metrics`, including `router_feedback_total{metric, model}` for feedback writes
- OpenTelemetry traces via FastAPI instrumentation
- Grafana dashboard from `dashboards/grafana.json`

## Extension Points

- Add a provider by implementing `BaseProviderAdapter` and registering it in `AdapterRegistry`.
- Add a strategy by subclassing `RoutingStrategy` and wiring it in `build_strategies`.
- Replace classifier weights in `LogisticComplexityClassifier` or load weights from `scripts/train_classifier.py` output.
