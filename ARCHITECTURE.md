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
