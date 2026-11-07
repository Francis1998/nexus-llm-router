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
- `health_check()`

Concrete adapters live in `src/adapters/`.

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

`src/api/main.py` exposes `/v1/chat/completions` and the OpenAI Responses API
(`/v1/responses`, translated by `src/api/responses.py`) over the same router.
Both endpoints share one ingress path: the `Authorization` header is reduced to
a non-reversible key fingerprint, `X-Tenant-Id` hard limits and
`Idempotency-Key` replay run before routing, and non-streaming requests use the
exact-match response cache. Router failures map to 429 / 402 / 503. Responses
streams pull the first non-empty chunk before emitting the SSE lifecycle so
failures before any output stay real HTTP errors; `NexusRouter.complete_stream`'s
`on_complete` callback supplies the final usage for `response.completed`.
`complete_stream` falls back to the next candidate only while no text has been
streamed; after that a provider failure ends the stream instead of splicing a
second answer onto the partial one.

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
