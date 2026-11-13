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
is never dispatched.

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
(adapter credentials and circuit-breaker state).

## Audit Log

`AuditLog` persists newline-delimited JSON to `NEXUS_AUDIT_LOG_PATH`. Each response records request id, chosen model, provider, strategy, rationale, latency, token cost, token usage, and final state.

## Observability

Nexus emits:

- Structured JSON logs through `structlog`
- Prometheus metrics from `/metrics`
- OpenTelemetry traces via FastAPI instrumentation
- Grafana dashboard from `dashboards/grafana.json`

## Extension Points

- Add a provider by implementing `BaseProviderAdapter` and registering it in `AdapterRegistry`.
- Add a strategy by subclassing `RoutingStrategy` and wiring it in `build_strategies`.
- Replace classifier weights in `LogisticComplexityClassifier` or load weights from `scripts/train_classifier.py` output.
