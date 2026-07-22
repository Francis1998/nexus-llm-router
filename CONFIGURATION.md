# Configuration

Nexus uses `pydantic-settings`. Application settings use the `NEXUS_` prefix. Provider settings use provider-native environment names.

## Core Settings

```dotenv
NEXUS_ENVIRONMENT=development
NEXUS_DEFAULT_STRATEGY=rule-based
NEXUS_AUDIT_LOG_PATH=migrations/audit-log.jsonl
NEXUS_BUDGET_CAP_USD=25.0
NEXUS_RATE_LIMIT_CAPACITY=120
NEXUS_RATE_LIMIT_REFILL_PER_SECOND=2.0
NEXUS_ENABLE_PII_SCRUBBING=false
NEXUS_QUALITY_FLOOR=0.72
```

## Provider Credentials

```dotenv
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
MOONSHOT_API_KEY=
MOONSHOT_BASE_URL=https://api.moonshot.ai/v1
REQUEST_TIMEOUT_SECONDS=30
```

## A/B Routing Settings

The `ab` strategy assigns a stable bucket from the request ID and compares two configured model arms without changing application code.

```dotenv
NEXUS_AB_MODEL_A=gpt-4.1-mini
NEXUS_AB_MODEL_B=claude-haiku-4-5
NEXUS_AB_MODEL_A_WEIGHT=0.5
```

Use `gpt-5.5`, `claude-sonnet-4-6`, `gemini-3.1-pro-preview`, or `kimi-k2` for higher-quality evaluation arms when the experiment budget allows it.

## Weighted-Blend Routing Settings

The `weighted-blend` strategy selects the model that maximizes a tunable
composite of normalized quality, cost, and rolling p95 latency (cost and latency
are min-max inverted, so cheaper and faster candidates score higher). Weights are
normalized to sum to one, so only their ratios matter; all-zero weights fall back
to pure quality.

```dotenv
NEXUS_BLEND_QUALITY_WEIGHT=0.5
NEXUS_BLEND_COST_WEIGHT=0.3
NEXUS_BLEND_LATENCY_WEIGHT=0.2
```

## Budget-Aware Routing Settings

The `budget-aware` strategy is the dual of `cost-optimal`: instead of minimizing
cost subject to a quality floor, it maximizes quality subject to a hard
per-request cost ceiling. It selects the highest-quality domain-eligible model
whose estimated request cost stays within `NEXUS_REQUEST_COST_CEILING_USD`, and
falls back to the cheapest eligible model when nothing fits the ceiling.

```dotenv
NEXUS_REQUEST_COST_CEILING_USD=0.05
```

## Sticky-Session Routing

The `sticky-session` strategy pins every request that shares a `session_id` to a
single model via consistent hashing over the domain-eligible candidates. This
keeps multi-turn conversations on one model (stable context handling and
provider prompt-cache affinity) while spreading distinct sessions roughly
uniformly across the eligible pool for session-level load balancing. It requires
no additional configuration; the pin is derived from the request's `session_id`.

## Value Routing

The `value` strategy selects the domain-eligible model with the highest
quality-per-dollar ratio (`quality_score` divided by estimated request cost).
Unlike `cost-optimal` (minimize cost subject to a quality *floor*) and
`budget-aware` (maximize quality subject to a cost *ceiling*), it needs no
threshold to tune: it maximizes spend *efficiency* directly, favouring models
whose quality justifies their price while still choosing a premium model when
nothing cheaper is close in quality. It requires no additional configuration.

## Canary Routing

The `canary` strategy supports progressive delivery: it routes a configurable
fraction of traffic to a new *canary* model while the rest stays on a proven
*stable* model, so a regression is caught on a slice of requests before a full
cutover. Bucketing is a stable hash of `request_id`, so a given request always
lands on the same arm. Unlike the symmetric `ab` strategy, canary routing is
**health-gated**: whenever the canary provider's circuit breaker is open, all
traffic is routed to the stable model so a failing canary cannot keep drawing
its share of live traffic.

```dotenv
NEXUS_CANARY_STABLE_MODEL=gpt-4.1-mini
NEXUS_CANARY_MODEL=gpt-5.5
NEXUS_CANARY_WEIGHT=0.1
```

`NEXUS_CANARY_WEIGHT` is the fraction of traffic (within `[0.0, 1.0]`) sent to
the canary model; both model names must exist in the catalog.

## Latency-Budget Routing

The `latency-budget` strategy is the latency-domain dual of `budget-aware`:
instead of maximizing quality subject to a cost ceiling, it maximizes quality
subject to a hard *latency* ceiling. It selects the highest-quality
domain-eligible model whose provider rolling p95 latency stays within
`NEXUS_LATENCY_SLA_MS`, so a request only trades quality for speed when the SLA
requires it (unlike `latency-aware`, which always minimizes latency). Providers
with no recorded latency yet are treated as within the SLA so a cold start still
routes to the best model; when no provider meets the SLA it falls back to the
lowest-p95 eligible model.

```dotenv
NEXUS_LATENCY_SLA_MS=750.0
```

`NEXUS_LATENCY_SLA_MS` is the maximum acceptable provider p95 latency per
request, in milliseconds (non-negative).

## Epsilon-Greedy Routing

The `epsilon-greedy` strategy is a classic bandit policy over the model catalog:
with probability `NEXUS_EPSILON` it *explores* by picking uniformly among
domain-eligible candidates (via a second stable hash of `request_id`), and
otherwise *exploits* by selecting the highest-`quality_score` eligible model.
Bucketing matches canary/A/B (`sha256(request_id)[:8] / 0xFFFFFFFF`), so a given
request always resolves to the same arm for replay and auditability while
distinct requests still explore at the configured rate. Useful when quality
priors are mostly trusted but you still want a small live sample across
GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2.

```dotenv
NEXUS_EPSILON=0.1
```

`NEXUS_EPSILON` is the explore probability within `[0.0, 1.0]` (default `0.1`).
See [docs/guides/EPSILON_GREEDY_GUIDE.md](docs/guides/EPSILON_GREEDY_GUIDE.md).

## Token-Budget Routing

The `token-budget` strategy maximizes quality subject to a hard token ceiling: it
selects the highest-quality domain-eligible model whose
`min(context_window, request.token_budget)` can hold
`prompt_tokens_estimate + max_tokens`. Useful for long RAG contexts across
GPT-5.5, Claude Sonnet 4.6, Gemini 2.5, and Kimi K2 without risking provider
context overflows. When no model fits it falls back to the largest-context
eligible candidate. Requires no additional env vars; set `token_budget` on the
request (default `4096`). See
[docs/guides/TOKEN_BUDGET_GUIDE.md](docs/guides/TOKEN_BUDGET_GUIDE.md).

## SLO-Aware Routing

The `slo-aware` strategy maximizes quality subject to a rolling availability SLO:
it selects the highest-quality domain-eligible model whose provider success rate
meets `NEXUS_AVAILABILITY_SLO` (default `0.99`). Useful when soft degradation
would otherwise keep routing to GPT-5.5 / Claude Sonnet 4.6 / Gemini 2.5 / Kimi
K2 providers that are burning error budget. Providers with no observations yet
are treated as healthy; when nothing meets the SLO it falls back to the highest
success-rate eligible model. See
[docs/guides/SLO_AWARE_GUIDE.md](docs/guides/SLO_AWARE_GUIDE.md).

```dotenv
NEXUS_AVAILABILITY_SLO=0.99
```
## Semantic-Cache Routing

The `semantic-cache` strategy is Portkey/LiteLLM-style cache-aware routing: when
`request.metadata.cache_hit` is truthy it prefers the cheapest domain-eligible
realtime model among GPT-5.5, Claude Sonnet 4.6, Gemini 2.5, and Kimi K2; on a
miss it falls through to `cost-optimal` under `NEXUS_QUALITY_FLOOR`. Useful when
an upstream semantic cache already resolved the answer and frontier spend would
be wasted. See
[docs/guides/SEMANTIC_CACHE_STRATEGY_GUIDE.md](docs/guides/SEMANTIC_CACHE_STRATEGY_GUIDE.md).

## Failover-Priority Routing

The `failover-priority` strategy is LiteLLM-style ordered failover: it walks
`NEXUS_FAILOVER_PRIORITY` (default GPT-5.5 → Claude Sonnet 4.6 → Gemini 2.5 →
Kimi K2) and selects the first model whose provider circuit is closed. Unhealthy
providers are skipped; when every preference is unhealthy it still routes to the
first listed catalog model. The fallback chain preserves the remaining priority
order. See
[docs/guides/FAILOVER_PRIORITY_GUIDE.md](docs/guides/FAILOVER_PRIORITY_GUIDE.md).

```dotenv
NEXUS_FAILOVER_PRIORITY=["gpt-5.5","claude-sonnet-4-6","gemini-3.1-pro-preview","kimi-k2"]
```

## Least-Busy Routing

The `least-busy` strategy selects the highest-quality domain-eligible model on
the provider with the lowest live in-flight load score. The router increments
the provider counter immediately before dispatch and decrements it in a
completion/failure cleanup path, so concurrent requests spread away from
currently saturated providers. If multiple providers have the same load, ties
prefer higher `quality_score`, then lower estimated request cost.

No additional `NEXUS_*` setting is required; select it with
`NEXUS_DEFAULT_STRATEGY=least-busy` or per request with
`X-Router-Strategy: least-busy`. See
[docs/guides/LEAST_BUSY_GUIDE.md](docs/guides/LEAST_BUSY_GUIDE.md).

## Per-Request Strategy Selection

Set `X-Router-Strategy` to one of:

- `rule-based`
- `classifier`
- `cost-optimal`
- `latency-aware`
- `reliability-aware`
- `weighted-blend`
- `budget-aware`
- `sticky-session`
- `value`
- `canary`
- `latency-budget`
- `complexity-tier`
- `round-robin`
- `cascade`
- `epsilon-greedy`
- `geo-region`
- `token-budget`
- `slo-aware`
- `semantic-cache`
- `least-busy`
- `failover-priority`
- `ab`

If the header is absent, Nexus uses `NEXUS_DEFAULT_STRATEGY`.

## Built-In Model Catalog

The default catalog lives in `src/router/config.py` and includes:

- OpenAI: `gpt-5.5`, `gpt-4.1-mini`
- Anthropic: `claude-sonnet-4-6`, `claude-haiku-4-5`
- Google: `gemini-3.1-pro-preview`, `gemini-3.5-flash`
- Moonshot: `kimi-k2`

Each model has quality, cost, domain, and realtime-support priors. Replace or extend the catalog when onboarding provider-specific SKUs.
