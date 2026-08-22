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
NEXUS_PROMPT_PREFIX_CACHE_MIN_CHARS=512
NEXUS_CONCURRENCY_CAP=8
NEXUS_TOKEN_BUCKET_CAPACITY=10
NEXUS_TOKEN_BUCKET_REFILL_PER_SEC=1.0
NEXUS_TIER_FRONTIER_RPM=30
NEXUS_TIER_MID_RPM=60
NEXUS_TIER_ECONOMY_RPM=120
NEXUS_TOKEN_RPM_CEILING=100000
NEXUS_TENANT_BUDGET_CASCADE_SOFT=10.0
NEXUS_TENANT_BUDGET_CASCADE_HARD=12.5
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

## Provider-Family Cost-Ceiling Routing Settings

The `provider-family-cost-ceiling` strategy applies OpenRouter/LiteLLM-style
spend ceilings per provider family (`openai`, `anthropic`, `google`,
`moonshot`). It selects the highest-quality domain-eligible model whose
estimated request cost stays within that family's ceiling (default from
`NEXUS_PROVIDER_FAMILY_COST_CEILING_USD` for every family), covering GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog priors. When no family has an
affordable eligible model it falls back across families to the cheapest
eligible candidate. Constructor-level `family_ceilings_usd` overrides can
tighten or relax one family without changing the default.

```dotenv
NEXUS_PROVIDER_FAMILY_COST_CEILING_USD=0.05
```

See
[docs/guides/PROVIDER_FAMILY_COST_CEILING_GUIDE.md](docs/guides/PROVIDER_FAMILY_COST_CEILING_GUIDE.md).

## Soft-Family-Budget Routing Settings

The `soft-family-budget` strategy tracks **observed** rolling spend per provider
family and deprioritizes families that have crossed a soft budget. Among families
still under budget it selects the highest-quality domain-eligible model; when
every family is hot it falls back to the cheapest eligible model from another
family. Spend is recorded by the engine after successful completions into shared
`FamilySpendWindow`.

```dotenv
NEXUS_SOFT_FAMILY_BUDGET_USD=5.0
NEXUS_SOFT_FAMILY_BUDGET_WINDOW_SECONDS=3600
```

`NEXUS_SOFT_FAMILY_BUDGET_USD` is the soft rolling spend ceiling per provider
family in USD (non-negative). `NEXUS_SOFT_FAMILY_BUDGET_WINDOW_SECONDS` is the
rolling window length in seconds (positive; default one hour).

See
[docs/guides/SOFT_FAMILY_BUDGET_GUIDE.md](docs/guides/SOFT_FAMILY_BUDGET_GUIDE.md).

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

## Canary-Tier-Blend Routing

The `canary-tier-blend` strategy combines progressive canary delivery with
complexity-tier affinity. On the canary slice it prefers the canary when it
matches the inferred frontier/mid/economy tier, otherwise still routes to the
canary when healthy. Off the canary slice, or when the canary provider circuit
is open, it prefers the highest-quality domain-eligible model in the target
tier and falls back to top quality when no tier match exists. It reuses the
same `NEXUS_CANARY_*` settings as `canary`.

```dotenv
NEXUS_CANARY_STABLE_MODEL=gpt-4.1-mini
NEXUS_CANARY_MODEL=gpt-5.5
NEXUS_CANARY_WEIGHT=0.1
```

See
[docs/guides/CANARY_TIER_BLEND_GUIDE.md](docs/guides/CANARY_TIER_BLEND_GUIDE.md).

## Shadow-Traffic-Mirror Routing

The `shadow-traffic-mirror` strategy selects a **primary** model like
`cost-optimal` (minimum estimated cost subject to `NEXUS_QUALITY_FLOOR`), then
on a deterministic `request_id` hash slice annotates the rationale with a
**shadow mirror** candidate: the highest-quality feasible model from a
different provider than the primary. The decide phase still returns one primary
model; audit trails capture the mirror for dual-run telemetry hooks.

```dotenv
NEXUS_SHADOW_TRAFFIC_PERCENT=5.0
NEXUS_QUALITY_FLOOR=0.72
```

`NEXUS_SHADOW_TRAFFIC_PERCENT` is the percentage of traffic (within
`[0.0, 100.0]`) whose rationale names a shadow mirror model.

See
[docs/guides/SHADOW_TRAFFIC_MIRROR_GUIDE.md](docs/guides/SHADOW_TRAFFIC_MIRROR_GUIDE.md).

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

## Latency-SLO-Shed Routing

The `latency-slo-shed` strategy enforces a softer latency service-level
objective than `latency-budget`. When at least one domain-eligible provider's
rolling p95 fits under `NEXUS_LATENCY_SLO_MS`, candidates above the SLO are
**shed** (excluded) and the highest-quality under-SLO model wins. When every
provider exceeds the SLO it falls back to the lowest-p95 eligible model.
Providers with no observations yet are treated as within the SLO.

```dotenv
NEXUS_LATENCY_SLO_MS=2000.0
```

`NEXUS_LATENCY_SLO_MS` is the maximum acceptable provider p95 latency per
request, in milliseconds (non-negative; default `2000`).

See
[docs/guides/LATENCY_SLO_SHED_GUIDE.md](docs/guides/LATENCY_SLO_SHED_GUIDE.md).


## Prompt-Length-Tier-Shed Routing

The `prompt-length-tier-shed` strategy sheds frontier-tier models when
`prompt_tokens_estimate` exceeds `NEXUS_PROMPT_LENGTH_TIER_TOKENS`. Long prompts
prefer the highest-quality mid/economy candidate; short prompts keep pure
quality ranking so frontier GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
models remain available for compact hard tasks.

```dotenv
NEXUS_PROMPT_LENGTH_TIER_TOKENS=8000
```

`NEXUS_PROMPT_LENGTH_TIER_TOKENS` is the prompt-token threshold that triggers
frontier shedding (integer >= 1; default `8000`).

See
[docs/guides/PROMPT_LENGTH_TIER_SHED_GUIDE.md](docs/guides/PROMPT_LENGTH_TIER_SHED_GUIDE.md).


## Retry-Budget-Aware-Failover Routing

The `retry-budget-aware-failover` strategy reads `metadata.retry_remaining`
(falling back to `NEXUS_RETRY_BUDGET_DEFAULT`) and prefers highest-quality
healthy models while budget remains. On the last attempt (`<= 1`) it failovers
to the lowest rolling-p95 healthy model so GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic still has a fast final try.

```dotenv
NEXUS_RETRY_BUDGET_DEFAULT=3
```

`NEXUS_RETRY_BUDGET_DEFAULT` is the default remaining retry count when metadata
omits `retry_remaining` (integer >= 0; default `3`).

See
[docs/guides/RETRY_BUDGET_AWARE_FAILOVER_GUIDE.md](docs/guides/RETRY_BUDGET_AWARE_FAILOVER_GUIDE.md).


## Cache-Hit-Sticky-Warm-Pool Routing

The `cache-hit-sticky-warm-pool` strategy pins identical long prompt prefixes
onto one domain-eligible model so provider prompt caches stay warm for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. Short prefixes fall back to
`session_id`. Unhealthy sticky primaries failover along a deterministic ring.

```dotenv
NEXUS_CACHE_HIT_STICKY_MIN_CHARS=64
```

`NEXUS_CACHE_HIT_STICKY_MIN_CHARS` is the minimum prefix length before sticky
hashing activates (integer >= 1; default `64`).

See
[docs/guides/CACHE_HIT_STICKY_WARM_POOL_GUIDE.md](docs/guides/CACHE_HIT_STICKY_WARM_POOL_GUIDE.md).

## Embedding-Cache-Key-Namespace Routing

The `embedding-cache-key-namespace` strategy pins sticky/cache routing under a
tenant-isolated namespace so embedding-cache keys do not collide across tenants
for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

```dotenv
NEXUS_EMBEDDING_CACHE_NAMESPACE_PREFIX=embed
```

`NEXUS_EMBEDDING_CACHE_NAMESPACE_PREFIX` is prepended to the tenant/session
scope before sticky hashing (non-empty string; default `embed`).

See
[docs/guides/EMBEDDING_CACHE_KEY_NAMESPACE_GUIDE.md](docs/guides/EMBEDDING_CACHE_KEY_NAMESPACE_GUIDE.md).


## Circuit-Breaker-Half-Open-Probe Routing

The `circuit-breaker-half-open-probe` strategy prefers healthy closed providers
and allows only limited concurrent probes into half-open/recovering providers
for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

```dotenv
NEXUS_CIRCUIT_HALF_OPEN_PROBE_BUDGET=2
```

`NEXUS_CIRCUIT_HALF_OPEN_PROBE_BUDGET` is the maximum concurrent in-flight
attempts across half-open providers before probes are deferred (integer >= 1;
default `2`).

See
[docs/guides/CIRCUIT_BREAKER_HALF_OPEN_PROBE_GUIDE.md](docs/guides/CIRCUIT_BREAKER_HALF_OPEN_PROBE_GUIDE.md).


## Tenant-Concurrency-Lease Routing

The `tenant-concurrency-lease` strategy prefers providers with remaining
per-tenant in-flight headroom so one noisy tenant cannot saturate GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 backends for everyone else.

```dotenv
NEXUS_TENANT_CONCURRENCY_LEASE=8
```

`NEXUS_TENANT_CONCURRENCY_LEASE` is the maximum concurrent in-flight attempts
per tenant/provider pair before that provider is skipped for the tenant
(integer >= 1; default `8`).

See
[docs/guides/TENANT_CONCURRENCY_LEASE_GUIDE.md](docs/guides/TENANT_CONCURRENCY_LEASE_GUIDE.md).

## Provider-Error-Budget-Shed Routing

The `provider-error-budget-shed` strategy prefers healthy domain-eligible
providers whose rolling provider error rate stays within budget. Error rate is
derived from shared `SuccessStats` as `1 - success_rate(provider)`, so providers
with no observations start at 0% error and remain eligible for cold-start
traffic across GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 backends.

```dotenv
NEXUS_PROVIDER_ERROR_BUDGET_RATE=0.15
```

`NEXUS_PROVIDER_ERROR_BUDGET_RATE` is the maximum rolling provider error rate
accepted for primary selection (float in `[0.0, 1.0]`; default `0.15`). Among
under-budget candidates the strategy selects highest quality; when every
eligible provider is over budget it falls back to lowest error rate, then
highest quality.

See
[docs/guides/PROVIDER_ERROR_BUDGET_SHED_GUIDE.md](docs/guides/PROVIDER_ERROR_BUDGET_SHED_GUIDE.md).

## Region-Latency-P99-Shed Routing

The `region-latency-p99-shed` strategy prefers domain-eligible models that match
the request region and whose rolling provider p99 latency stays under threshold.
Providers above `NEXUS_REGION_LATENCY_P99_MS` are shed when faster regional
alternatives exist, so GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
traffic avoids hot regional tails.

```dotenv
NEXUS_REGION_LATENCY_P99_MS=3000.0
```

`NEXUS_REGION_LATENCY_P99_MS` is the maximum rolling provider p99 latency
accepted for primary selection (float >= 0; default `3000`). Among under-
threshold regional candidates the strategy selects highest quality; when every
regional provider is over threshold it falls back to lowest p99, then highest
quality.

See
[docs/guides/REGION_LATENCY_P99_SHED_GUIDE.md](docs/guides/REGION_LATENCY_P99_SHED_GUIDE.md).

## Sticky-Canary-Cost Routing

The `sticky-canary-cost` strategy pins tenants to a primary model via consistent
hashing, then blends a deterministic explore slice toward cheaper healthy
models. Off-slice GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic
keeps sticky affinity with healthy ring failover.

```dotenv
NEXUS_STICKY_CANARY_COST_PERCENT=10.0
```

`NEXUS_STICKY_CANARY_COST_PERCENT` is the percentage of traffic (within
`[0.0, 100.0]`; default `10`) that explores a strictly cheaper healthy model
than the sticky primary. When no cheaper healthy option exists the explore
slice still returns the sticky primary.

See
[docs/guides/STICKY_CANARY_COST_GUIDE.md](docs/guides/STICKY_CANARY_COST_GUIDE.md).



## Queue-Depth-Fairness Routing

The `queue-depth-fairness` strategy prefers domain-eligible providers whose live
`InflightStats` queue depth stays under a soft fairness cap. Providers at or
above the cap are shed when shallower alternatives exist, so GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic shares capacity fairly instead
of concentrating on one deep queue.

```dotenv
NEXUS_QUEUE_DEPTH_SOFT_CAP=4
```

`NEXUS_QUEUE_DEPTH_SOFT_CAP` is the soft maximum in-flight depth per provider
accepted for primary selection (integer >= 1; default `4`). Among under-cap
candidates the strategy selects highest quality; when every eligible provider is
at or above the cap it falls back to lowest depth, then highest quality.

See
[docs/guides/QUEUE_DEPTH_FAIRNESS_GUIDE.md](docs/guides/QUEUE_DEPTH_FAIRNESS_GUIDE.md).


## Provider-Quota-Fair-Share Routing
- `token-bucket-tenant`

The `provider-quota-fair-share` strategy keeps a bounded local window of recent
provider selections and computes equal share across providers eligible for the
current request domain. Providers at or above equal share are shed whenever an
under-share alternative exists; quality and estimated cost break ties within the
preferred pool for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

```dotenv
NEXUS_PROVIDER_QUOTA_LOOKBACK=100
```

`NEXUS_PROVIDER_QUOTA_LOOKBACK` is the number of recent routing decisions
retained by each router process (integer >= 1; default `100`). Lower values react
faster; higher values smooth bursts. See
[docs/guides/PROVIDER_QUOTA_FAIR_SHARE_GUIDE.md](docs/guides/PROVIDER_QUOTA_FAIR_SHARE_GUIDE.md).


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

## Adaptive-Exploration Routing

The `adaptive-exploration` strategy is a decaying epsilon-greedy bandit over the
model catalog. It starts at `NEXUS_ADAPTIVE_EXPLORATION_BASE` explore probability
while `SuccessStats` are empty, then decays toward
`NEXUS_ADAPTIVE_EXPLORATION_MIN` as total provider successes grow:

`epsilon = min + (base - min) / (1 + total_successes)`.

Explore/exploit bucketing matches canary/A/B/`epsilon-greedy`, so a given
request stays replayable while early traffic still samples GPT-5.5, Claude Sonnet
4.6, Gemini 3.x, and Kimi K2 more broadly than a fixed epsilon.

```dotenv
NEXUS_ADAPTIVE_EXPLORATION_BASE=0.2
NEXUS_ADAPTIVE_EXPLORATION_MIN=0.02
```

Both values are explore probabilities within `[0.0, 1.0]`; `min` must be `<=`
`base`. Defaults are `0.2` → `0.02`. See
[docs/guides/ADAPTIVE_EXPLORATION_GUIDE.md](docs/guides/ADAPTIVE_EXPLORATION_GUIDE.md).

## Sticky-Region-Failover Routing

The `sticky-region-failover` strategy combines geo-region preference with
session stickiness and ordered failover. It walks an ordered region list (request
`region` first, then `NEXUS_STICKY_REGION_FAILOVER_PREFERENCES`), selects the
first region with at least one healthy domain-eligible model, and pins
`session_id` to one model in that pool via consistent hashing. When the
preferred region has no healthy providers it advances to the next region while
keeping sticky affinity inside the active pool.

```dotenv
NEXUS_STICKY_REGION_FAILOVER_PREFERENCES=["eu","us","cn","global"]
```

`NEXUS_STICKY_REGION_FAILOVER_PREFERENCES` is the ordered failover list used
when a request omits `region`. Request `region` is always tried first.

See
[docs/guides/STICKY_REGION_FAILOVER_GUIDE.md](docs/guides/STICKY_REGION_FAILOVER_GUIDE.md).

## Sticky-Tenant-Hash Routing

The `sticky-tenant-hash` strategy pins each tenant to a primary model using
consistent hashing on `metadata.tenant_id` (then `metadata.user_id`,
`metadata.sticky_key`, top-level `user_id`, and finally `session_id`). Unlike
`sticky-session`, which hashes only `session_id` for multi-turn conversational
affinity, this strategy keeps GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
traffic stable per customer across sessions. When the sticky primary provider is
unhealthy the strategy walks a deterministic ring to the next healthy candidate.

No additional `NEXUS_*` environment variables are required. Select the strategy
via `NEXUS_DEFAULT_STRATEGY=sticky-tenant-hash` or the `X-Router-Strategy`
header and pass `metadata.tenant_id` on each request.

See
[docs/guides/STICKY_TENANT_HASH_GUIDE.md](docs/guides/STICKY_TENANT_HASH_GUIDE.md).

## Multi-Region-Latency-Hedge Routing

The `multi-region-latency-hedge` strategy keeps GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic on the highest-quality model in the request's
primary region (`request.region`, default `global`). When that model's provider
rolling **p50** exceeds `NEXUS_LATENCY_HEDGE_MS`, and at least one
secondary-region candidate exists, it hedges to the lowest-p50 secondary model.
Otherwise it stays on the primary quality preference. Providers with no
observations yet report `0.0` p50 and keep primary routing.

```dotenv
NEXUS_LATENCY_HEDGE_MS=500.0
```

`NEXUS_LATENCY_HEDGE_MS` is the primary-region provider p50 threshold in
milliseconds that triggers hedging (non-negative; default `500`).

See
[docs/guides/MULTI_REGION_LATENCY_HEDGE_GUIDE.md](docs/guides/MULTI_REGION_LATENCY_HEDGE_GUIDE.md).

## Token-Budget Routing

The `token-budget` strategy maximizes quality subject to a hard token ceiling: it
selects the highest-quality domain-eligible model whose
`min(context_window, request.token_budget)` can hold
`prompt_tokens_estimate + max_tokens`. Useful for long RAG contexts across
GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2 without risking provider
context overflows. When no model fits it falls back to the largest-context
eligible candidate. Requires no additional env vars; set `token_budget` on the
request (default `4096`). See
[docs/guides/TOKEN_BUDGET_GUIDE.md](docs/guides/TOKEN_BUDGET_GUIDE.md).

## SLO-Aware Routing

The `slo-aware` strategy maximizes quality subject to a rolling availability SLO:
it selects the highest-quality domain-eligible model whose provider success rate
meets `NEXUS_AVAILABILITY_SLO` (default `0.99`). Useful when soft degradation
would otherwise keep routing to GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi
K2 providers that are burning error budget. Providers with no observations yet
are treated as healthy; when nothing meets the SLO it falls back to the highest
success-rate eligible model. See
[docs/guides/SLO_AWARE_GUIDE.md](docs/guides/SLO_AWARE_GUIDE.md).

```dotenv
NEXUS_AVAILABILITY_SLO=0.99
```

## Adaptive-Timeout Routing

The `adaptive-timeout` strategy combines the latency-budget and SLO-aware signal
stores: it derives a timeout budget from the request latency requirement,
provider rolling p95 latency, and recent success/error signals. Realtime
requests use `NEXUS_LATENCY_SLA_MS` as the base budget, batch requests receive a
wider budget, and provider failures inflate the effective latency. Useful when
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic should prefer faster
models during provider latency spikes but keep higher-quality slower models when
the budget is comfortable. See
[docs/guides/ADAPTIVE_TIMEOUT_GUIDE.md](docs/guides/ADAPTIVE_TIMEOUT_GUIDE.md).

```dotenv
NEXUS_DEFAULT_STRATEGY=adaptive-timeout
NEXUS_LATENCY_SLA_MS=750
```

## Adaptive-Timeout-Hedge Routing

The `adaptive-timeout-hedge` strategy starts from the highest-quality
domain-eligible model, then compares its rolling provider p95 with an adaptive
threshold derived from the fastest positive p95 in shared `LatencyStats`.
When the quality leader exceeds
`fastest_p95 * NEXUS_ADAPTIVE_TIMEOUT_HEDGE_RATIO`, Nexus hedges to the fastest
observed provider alternative. Unknown (`0.0ms`) latency is excluded from hedge
targets.

```dotenv
NEXUS_DEFAULT_STRATEGY=adaptive-timeout-hedge
NEXUS_ADAPTIVE_TIMEOUT_HEDGE_RATIO=1.5
```

`NEXUS_ADAPTIVE_TIMEOUT_HEDGE_RATIO` must be at least `1.0` (default `1.5`).
This strategy is distinct from `adaptive-timeout`'s urgency/error-risk budget
and `multi-region-latency-hedge`'s fixed p50 regional threshold. See
[docs/guides/ADAPTIVE_TIMEOUT_HEDGE_GUIDE.md](docs/guides/ADAPTIVE_TIMEOUT_HEDGE_GUIDE.md).

## Token-Bucket-Tenant Routing

The `token-bucket-tenant` strategy maintains an independent local request-token
bucket for each `metadata.tenant_id` (with metadata user/sticky, top-level user,
and session fallbacks). Each routing decision consumes one quota token.
In-budget requests select the highest-quality domain-eligible model; over-budget
requests remain available but shed to the cheapest eligible GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 model.

```dotenv
NEXUS_DEFAULT_STRATEGY=token-bucket-tenant
NEXUS_TOKEN_BUCKET_TENANT_RATE=5.0
```

`NEXUS_TOKEN_BUCKET_TENANT_RATE` is the positive quota-token refill rate per
tenant per second (default `5.0`). Capacity equals one second of refill, with a
minimum one-request burst. This is distinct from `token-bucket-burst`, whose
buckets represent provider capacity. See
[docs/guides/TOKEN_BUCKET_TENANT_GUIDE.md](docs/guides/TOKEN_BUCKET_TENANT_GUIDE.md).



## Region-Carbon-Blend Routing

The `region-carbon-blend` strategy blends inverse-normalized carbon intensity
with rolling provider p95 latency. Intensity comes from
`carbon_intensity:<provider>` metadata or regional heuristics (`eu`/`us`/`cn`/
`global`). `NEXUS_REGION_CARBON_BLEND_WEIGHT` defaults to `0.5` (`0` = latency
only, `1` = carbon only) for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic.

```dotenv
NEXUS_DEFAULT_STRATEGY=region-carbon-blend
NEXUS_REGION_CARBON_BLEND_WEIGHT=0.5
```

See
[docs/guides/REGION_CARBON_BLEND_GUIDE.md](docs/guides/REGION_CARBON_BLEND_GUIDE.md).


## Provider-Weight-Decay Routing

The `provider-weight-decay` strategy multiplies each provider's selection weight
by model quality. Failures multiply weight by
`NEXUS_PROVIDER_WEIGHT_DECAY_FACTOR` (default `0.5`); successes add
`NEXUS_PROVIDER_WEIGHT_RECOVER` (default `0.1`, capped at `1.0`). Cold-start
weights are `1.0`. Targets GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic.

```dotenv
NEXUS_DEFAULT_STRATEGY=provider-weight-decay
NEXUS_PROVIDER_WEIGHT_DECAY_FACTOR=0.5
NEXUS_PROVIDER_WEIGHT_RECOVER=0.1
```

See
[docs/guides/PROVIDER_WEIGHT_DECAY_GUIDE.md](docs/guides/PROVIDER_WEIGHT_DECAY_GUIDE.md).


## Retry-After-Respect Routing

The `retry-after-respect` strategy skips providers still inside a Retry-After
cooldown window and prefers the highest-quality healthy ready provider. When
every healthy provider is cooling it falls back to the next healthy provider
with the soonest remaining wait. The engine records cooldowns on rate-limit
failures using the response `Retry-After` value when present, otherwise
`NEXUS_RETRY_AFTER_DEFAULT_SECONDS` (default `30`) for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

```dotenv
NEXUS_DEFAULT_STRATEGY=retry-after-respect
NEXUS_RETRY_AFTER_DEFAULT_SECONDS=30
```

See
[docs/guides/RETRY_AFTER_RESPECT_GUIDE.md](docs/guides/RETRY_AFTER_RESPECT_GUIDE.md).


## Latency-Slope-Shed Routing

The `latency-slope-shed` strategy tracks a short per-provider latency sample
window (`NEXUS_LATENCY_SLOPE_WINDOW`, default `10`) and computes an EWMA slope
in ms per sample step. When the quality leader's slope exceeds
`NEXUS_LATENCY_SLOPE_THRESHOLD_MS` (default `25`), traffic sheds to a
lower-latency / cheaper healthy alternative for GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.

```dotenv
NEXUS_DEFAULT_STRATEGY=latency-slope-shed
NEXUS_LATENCY_SLOPE_WINDOW=10
NEXUS_LATENCY_SLOPE_THRESHOLD_MS=25.0
```

See
[docs/guides/LATENCY_SLOPE_SHED_GUIDE.md](docs/guides/LATENCY_SLOPE_SHED_GUIDE.md).

## Provider-Hourly-Cost-Ceiling Routing

The `provider-hourly-cost-ceiling` strategy tracks rolling hourly estimated
spend per provider and hard-skips providers already over
`NEXUS_PROVIDER_HOURLY_COST_CEILING_USD` (default `5.0`). Distinct from
`provider-family-cost-ceiling` (per-request family ceilings). Among
under-ceiling providers it selects highest quality for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2; when every provider is over it
falls back to the cheapest eligible model.

```dotenv
NEXUS_DEFAULT_STRATEGY=provider-hourly-cost-ceiling
NEXUS_PROVIDER_HOURLY_COST_CEILING_USD=5.0
```

See
[docs/guides/PROVIDER_HOURLY_COST_CEILING_GUIDE.md](docs/guides/PROVIDER_HOURLY_COST_CEILING_GUIDE.md).

## Token-RPM-Ceiling Routing

The `token-rpm-ceiling` strategy tracks estimated prompt tokens per provider in
a rolling 60-second window. Before routing a request, it adds
`prompt_tokens_estimate` to each provider's current total and sheds providers
whose projected total would exceed the ceiling. The next highest-quality
domain-eligible provider receives the request; if every provider is projected
over, the least-loaded provider is selected. This protects GPT-5.5 / Claude
Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic from token-per-minute throttles.

```dotenv
NEXUS_DEFAULT_STRATEGY=token-rpm-ceiling
NEXUS_TOKEN_RPM_CEILING=100000
```

See
[docs/guides/TOKEN_RPM_CEILING_GUIDE.md](docs/guides/TOKEN_RPM_CEILING_GUIDE.md).

## Quality-Weighted-Sticky Routing

The `quality-weighted-sticky` strategy pins `session_id` via consistent hashing
like `sticky-session`, but allocates hash-ring share proportional to each
candidate's `quality_score`. Higher-quality models (for example Claude Sonnet
4.6 / GPT-5.5) absorb more sticky sessions while Gemini 3.x / Kimi K2 arms
still receive some. No extra `NEXUS_*` knobs — weights come from catalog
quality priors. Distinct from `sticky-tenant-hash`.

```dotenv
NEXUS_DEFAULT_STRATEGY=quality-weighted-sticky
```

See
[docs/guides/QUALITY_WEIGHTED_STICKY_GUIDE.md](docs/guides/QUALITY_WEIGHTED_STICKY_GUIDE.md).

## Semantic-Cache Routing

The `semantic-cache` strategy is Portkey/LiteLLM-style cache-aware routing: when
`request.metadata.cache_hit` is truthy it prefers the cheapest domain-eligible
realtime model among GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2; on a
miss it falls through to `cost-optimal` under `NEXUS_QUALITY_FLOOR`. Useful when
an upstream semantic cache already resolved the answer and frontier spend would
be wasted. See
[docs/guides/SEMANTIC_CACHE_STRATEGY_GUIDE.md](docs/guides/SEMANTIC_CACHE_STRATEGY_GUIDE.md).

## Failover-Priority Routing

The `failover-priority` strategy is LiteLLM-style ordered failover: it walks
`NEXUS_FAILOVER_PRIORITY` (default GPT-5.5 → Claude Sonnet 4.6 → Gemini 3.x →
Kimi K2) and selects the first model whose provider circuit is closed. Unhealthy
providers are skipped; when every preference is unhealthy it still routes to the
first listed catalog model. The fallback chain preserves the remaining priority
order. See
[docs/guides/FAILOVER_PRIORITY_GUIDE.md](docs/guides/FAILOVER_PRIORITY_GUIDE.md).

```dotenv
NEXUS_FAILOVER_PRIORITY=["gpt-5.5","claude-sonnet-4-6","gemini-3.1-pro-preview","kimi-k2"]
```

## Provider-Health Score Blend Routing
The `provider-health-score-blend` strategy is LiteLLM/Portkey-style
health-aware routing for the default GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 catalog mix. It scores domain-eligible models by blending rolling
provider success rate, inverse normalized provider p95 latency, model
`quality_score`, and inverse normalized estimated request cost. Provider circuit
availability is a hard gate: when any candidate's circuit is closed, open
circuits are excluded from primary scoring; when every circuit is open, Nexus
still returns the best scored model so decide-time remains deterministic.
```dotenv
NEXUS_HEALTH_BLEND_SUCCESS_WEIGHT=0.35
NEXUS_HEALTH_BLEND_LATENCY_WEIGHT=0.25
NEXUS_HEALTH_BLEND_QUALITY_WEIGHT=0.25
NEXUS_HEALTH_BLEND_COST_WEIGHT=0.15
```
Weights are non-negative and normalized to sum to one, so only ratios matter.
All-zero weights fall back to pure quality. See
[docs/guides/PROVIDER_HEALTH_SCORE_BLEND_GUIDE.md](docs/guides/PROVIDER_HEALTH_SCORE_BLEND_GUIDE.md).

## Health/Cost/Latency Routing
The `health-cost-latency` strategy is LiteLLM/Portkey-style ternary routing for
the default GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog mix. It
scores domain-eligible models by blending rolling provider success rate, inverse
normalized estimated request cost, and inverse normalized rolling provider p95
latency — without circuit-breaker gating or a separate quality component.
```dotenv
NEXUS_HCL_HEALTH_WEIGHT=0.4
NEXUS_HCL_COST_WEIGHT=0.3
NEXUS_HCL_LATENCY_WEIGHT=0.3
```
Weights are non-negative and normalized to sum to one, so only ratios matter.
All-zero weights fall back to pure health (success rate). See
[docs/guides/HEALTH_COST_LATENCY_GUIDE.md](docs/guides/HEALTH_COST_LATENCY_GUIDE.md).

## Soft-Rate-Limit Routing
The `soft-rate-limit` strategy is LiteLLM/Portkey-style soft backoff for
providers that recently returned 429 or rate-limit shaped errors. It prefers
healthy domain-eligible providers with fewer recent rate-limit observations,
then breaks ties by quality and estimated request cost. Successful attempts add
non-rate-limit observations, so GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi
K2 traffic can return to a provider after quota pressure cools down. No
additional `NEXUS_*` setting is required; select it with
`NEXUS_DEFAULT_STRATEGY=soft-rate-limit` or per request with
`X-Router-Strategy: soft-rate-limit`. See
[docs/guides/SOFT_RATE_LIMIT_GUIDE.md](docs/guides/SOFT_RATE_LIMIT_GUIDE.md).
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
## Prompt-Prefix-Cache Routing
The `prompt-prefix-cache` strategy gives long reusable system prompts sticky
provider/model affinity for OpenRouter/LiteLLM-style prompt caching. It hashes
the first `NEXUS_PROMPT_PREFIX_CACHE_MIN_CHARS` characters of joined `system`
messages and buckets that prefix across domain-eligible candidates, so requests
sharing a long prefix keep hitting the same GPT-5.5 / Claude Sonnet 4.6 / Gemini
3.x / Kimi K2 provider/model cache. Requests without a sufficiently long system
prompt fall back to `cost-optimal` under `NEXUS_QUALITY_FLOOR`.
NEXUS_PROMPT_PREFIX_CACHE_MIN_CHARS=512
See
[docs/guides/PROMPT_PREFIX_CACHE_STRATEGY_GUIDE.md](docs/guides/PROMPT_PREFIX_CACHE_STRATEGY_GUIDE.md).
## Cost/Latency Pareto Routing
The `cost-latency-pareto` strategy keeps only non-dominated domain-eligible
candidates under estimated request cost and rolling provider p95 latency, then
breaks ties by higher `quality_score` (then lower cost, lower latency, and model
name). Cold providers start at p95 `0.0`, so equal-latency startups collapse to
the cheapest frontier before the quality tie-break. No additional `NEXUS_*`
setting is required; select it with `NEXUS_DEFAULT_STRATEGY=cost-latency-pareto`
or per request with `X-Router-Strategy: cost-latency-pareto`. See
[docs/guides/COST_LATENCY_PARETO_GUIDE.md](docs/guides/COST_LATENCY_PARETO_GUIDE.md).

## Concurrency-Cap Routing
The `concurrency-cap` strategy is a LiteLLM/Portkey-style provider saturation
guard for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. It reads
the same live `InflightStats` counters as `least-busy`, skips providers whose
current in-flight count is at or above `NEXUS_CONCURRENCY_CAP`, and selects the
highest-quality domain-eligible model among the remaining providers. If every
eligible provider is already capped, it degrades deterministically to the
least-loaded eligible provider and records that condition in the rationale.

```dotenv
NEXUS_DEFAULT_STRATEGY=concurrency-cap
NEXUS_CONCURRENCY_CAP=8
```

`NEXUS_CONCURRENCY_CAP` is the maximum live attempts per provider before new
primary traffic is steered elsewhere (minimum `1`). See
[docs/guides/CONCURRENCY_CAP_GUIDE.md](docs/guides/CONCURRENCY_CAP_GUIDE.md).

## Token-Bucket-Burst Routing
The `token-bucket-burst` strategy is LiteLLM/Portkey/OpenRouter-style bursty quota
routing for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. Each
provider keeps a shared token bucket that refills over time. Primary selection
prefers providers with at least one available token, then breaks ties by quality
and estimated request cost. When every bucket is empty, Nexus falls back to the
highest remaining token fraction, then to the cheapest eligible model, and still
consumes one token from the chosen provider so buckets drain gradually instead
of hard-blocking.

```dotenv
NEXUS_DEFAULT_STRATEGY=token-bucket-burst
NEXUS_TOKEN_BUCKET_CAPACITY=10
NEXUS_TOKEN_BUCKET_REFILL_PER_SEC=1.0
```

`NEXUS_TOKEN_BUCKET_CAPACITY` is the per-provider burst ceiling (minimum `1`).
`NEXUS_TOKEN_BUCKET_REFILL_PER_SEC` is the refill rate in tokens per second and
must be positive. See
[docs/guides/TOKEN_BUCKET_BURST_GUIDE.md](docs/guides/TOKEN_BUCKET_BURST_GUIDE.md).

## Model-Tier-Rate-Limit Routing
The `model-tier-rate-limit` strategy is LiteLLM/Portkey-style tier-aware soft RPM
routing for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. It
infers each candidate's frontier/mid/economy tier from its model name, tracks
rolling per-provider request timestamps, and prefers providers still under the
tier-specific RPM ceiling. When every eligible provider is saturated it falls
back to the least-saturated provider, then quality and estimated cost.

```dotenv
NEXUS_DEFAULT_STRATEGY=model-tier-rate-limit
NEXUS_TIER_FRONTIER_RPM=30
NEXUS_TIER_MID_RPM=60
NEXUS_TIER_ECONOMY_RPM=120
```

`NEXUS_TIER_FRONTIER_RPM` defaults to `30`, `NEXUS_TIER_MID_RPM` to `60`, and
`NEXUS_TIER_ECONOMY_RPM` to `120`. Each must be at least `1`. See
[docs/guides/MODEL_TIER_RATE_LIMIT_GUIDE.md](docs/guides/MODEL_TIER_RATE_LIMIT_GUIDE.md).

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
- `adaptive-exploration`
- `geo-region`
- `region-tier-affinity`
- `token-budget`
- `slo-aware`
- `semantic-cache`
- `least-busy`
- `prompt-prefix-cache`
- `concurrency-cap`
- `soft-rate-limit`
- `cost-latency-pareto`
- `token-bucket-burst`
- `model-tier-rate-limit`
- `failover-priority`
- `provider-health-score-blend`
- `health-cost-latency`
- `embedding-cache-key-namespace`
- `circuit-breaker-half-open-probe`
- `provider-quota-fair-share`
- `adaptive-timeout-hedge`
- `token-bucket-tenant`
- `region-carbon-blend`
- `provider-weight-decay`
- `retry-after-respect`
- `latency-slope-shed`
- `provider-hourly-cost-ceiling`
- `quality-weighted-sticky`
- `provider-cold-start-bias`
- `tenant-fair-queue`
- `sticky-region-drain`
- `provider-canary-shadow-split`
- `ab`

If the header is absent, Nexus uses `NEXUS_DEFAULT_STRATEGY`.

## Built-In Model Catalog

The default catalog lives in `src/router/config.py` and includes:

- OpenAI: `gpt-5.5`, `gpt-4.1-mini`
- Anthropic: `claude-sonnet-4-6`, `claude-haiku-4-5`
- Google: `gemini-3.1-pro-preview`, `gemini-3.5-flash`
- Moonshot: `kimi-k2`

Each model has quality, cost, domain, and realtime-support priors. Replace or extend the catalog when onboarding provider-specific SKUs.

## Canary-Cost-Blend Routing
The `canary-cost-blend` strategy is a **cost-first** counterpart to
NEXUS_CANARY_COST_BLEND_PERCENT=10.0
`NEXUS_CANARY_COST_BLEND_PERCENT` is the percentage of traffic (within
[docs/guides/CANARY_COST_BLEND_GUIDE.md](docs/guides/CANARY_COST_BLEND_GUIDE.md).

## Token-Cost-Anomaly-Shed Routing

The `token-cost-anomaly-shed` strategy tracks a rolling mean cost-per-1k-tokens
via shared `CostAnomalyStats` (populated by the engine after successful
completions). It ranks domain-eligible candidates by quality and, when the top
pick's projected cost/1k exceeds `mean * NEXUS_TOKEN_COST_ANOMALY_RATIO`,
sheds to the cheapest healthy alternative below that cost. When no cheaper
healthy option exists it falls back to pure quality ranking. Cold start (empty
stats) routes to the highest-quality eligible model.

```dotenv
NEXUS_TOKEN_COST_ANOMALY_RATIO=2.0
```

`NEXUS_TOKEN_COST_ANOMALY_RATIO` is the positive multiplier above the rolling
mean cost/1k that triggers shedding (default `2.0`).

See
[docs/guides/TOKEN_COST_ANOMALY_SHED_GUIDE.md](docs/guides/TOKEN_COST_ANOMALY_SHED_GUIDE.md).

## Semantic-cache TTL affinity

`NEXUS_SEMANTIC_CACHE_TTL_SECONDS` (default `300.0`) is the warm TTL window for semantic-cache-ttl-affinity.

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
NEXUS_PROMPT_PREFIX_CACHE_MIN_CHARS=512
NEXUS_CONCURRENCY_CAP=8
NEXUS_TOKEN_BUCKET_CAPACITY=10
NEXUS_TOKEN_BUCKET_REFILL_PER_SEC=1.0
NEXUS_TIER_FRONTIER_RPM=30
NEXUS_TIER_MID_RPM=60
NEXUS_TIER_ECONOMY_RPM=120
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

## Provider-Family Cost-Ceiling Routing Settings

The `provider-family-cost-ceiling` strategy applies OpenRouter/LiteLLM-style
spend ceilings per provider family (`openai`, `anthropic`, `google`,
`moonshot`). It selects the highest-quality domain-eligible model whose
estimated request cost stays within that family's ceiling (default from
`NEXUS_PROVIDER_FAMILY_COST_CEILING_USD` for every family), covering GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog priors. When no family has an
affordable eligible model it falls back across families to the cheapest
eligible candidate. Constructor-level `family_ceilings_usd` overrides can
tighten or relax one family without changing the default.

```dotenv
NEXUS_PROVIDER_FAMILY_COST_CEILING_USD=0.05
```

See
[docs/guides/PROVIDER_FAMILY_COST_CEILING_GUIDE.md](docs/guides/PROVIDER_FAMILY_COST_CEILING_GUIDE.md).

## Soft-Family-Budget Routing Settings

The `soft-family-budget` strategy tracks **observed** rolling spend per provider
family and deprioritizes families that have crossed a soft budget. Among families
still under budget it selects the highest-quality domain-eligible model; when
every family is hot it falls back to the cheapest eligible model from another
family. Spend is recorded by the engine after successful completions into shared
`FamilySpendWindow`.

```dotenv
NEXUS_SOFT_FAMILY_BUDGET_USD=5.0
NEXUS_SOFT_FAMILY_BUDGET_WINDOW_SECONDS=3600
```

`NEXUS_SOFT_FAMILY_BUDGET_USD` is the soft rolling spend ceiling per provider
family in USD (non-negative). `NEXUS_SOFT_FAMILY_BUDGET_WINDOW_SECONDS` is the
rolling window length in seconds (positive; default one hour).

See
[docs/guides/SOFT_FAMILY_BUDGET_GUIDE.md](docs/guides/SOFT_FAMILY_BUDGET_GUIDE.md).

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

## Canary-Tier-Blend Routing

The `canary-tier-blend` strategy combines progressive canary delivery with
complexity-tier affinity. On the canary slice it prefers the canary when it
matches the inferred frontier/mid/economy tier, otherwise still routes to the
canary when healthy. Off the canary slice, or when the canary provider circuit
is open, it prefers the highest-quality domain-eligible model in the target
tier and falls back to top quality when no tier match exists. It reuses the
same `NEXUS_CANARY_*` settings as `canary`.

```dotenv
NEXUS_CANARY_STABLE_MODEL=gpt-4.1-mini
NEXUS_CANARY_MODEL=gpt-5.5
NEXUS_CANARY_WEIGHT=0.1
```

See
[docs/guides/CANARY_TIER_BLEND_GUIDE.md](docs/guides/CANARY_TIER_BLEND_GUIDE.md).

## Shadow-Traffic-Mirror Routing

The `shadow-traffic-mirror` strategy selects a **primary** model like
`cost-optimal` (minimum estimated cost subject to `NEXUS_QUALITY_FLOOR`), then
on a deterministic `request_id` hash slice annotates the rationale with a
**shadow mirror** candidate: the highest-quality feasible model from a
different provider than the primary. The decide phase still returns one primary
model; audit trails capture the mirror for dual-run telemetry hooks.

```dotenv
NEXUS_SHADOW_TRAFFIC_PERCENT=5.0
NEXUS_QUALITY_FLOOR=0.72
```

`NEXUS_SHADOW_TRAFFIC_PERCENT` is the percentage of traffic (within
`[0.0, 100.0]`) whose rationale names a shadow mirror model.

See
[docs/guides/SHADOW_TRAFFIC_MIRROR_GUIDE.md](docs/guides/SHADOW_TRAFFIC_MIRROR_GUIDE.md).

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

## Latency-SLO-Shed Routing

The `latency-slo-shed` strategy enforces a softer latency service-level
objective than `latency-budget`. When at least one domain-eligible provider's
rolling p95 fits under `NEXUS_LATENCY_SLO_MS`, candidates above the SLO are
**shed** (excluded) and the highest-quality under-SLO model wins. When every
provider exceeds the SLO it falls back to the lowest-p95 eligible model.
Providers with no observations yet are treated as within the SLO.

```dotenv
NEXUS_LATENCY_SLO_MS=2000.0
```

`NEXUS_LATENCY_SLO_MS` is the maximum acceptable provider p95 latency per
request, in milliseconds (non-negative; default `2000`).

See
[docs/guides/LATENCY_SLO_SHED_GUIDE.md](docs/guides/LATENCY_SLO_SHED_GUIDE.md).


## Prompt-Length-Tier-Shed Routing

The `prompt-length-tier-shed` strategy sheds frontier-tier models when
`prompt_tokens_estimate` exceeds `NEXUS_PROMPT_LENGTH_TIER_TOKENS`. Long prompts
prefer the highest-quality mid/economy candidate; short prompts keep pure
quality ranking so frontier GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
models remain available for compact hard tasks.

```dotenv
NEXUS_PROMPT_LENGTH_TIER_TOKENS=8000
```

`NEXUS_PROMPT_LENGTH_TIER_TOKENS` is the prompt-token threshold that triggers
frontier shedding (integer >= 1; default `8000`).

See
[docs/guides/PROMPT_LENGTH_TIER_SHED_GUIDE.md](docs/guides/PROMPT_LENGTH_TIER_SHED_GUIDE.md).


## Retry-Budget-Aware-Failover Routing

The `retry-budget-aware-failover` strategy reads `metadata.retry_remaining`
(falling back to `NEXUS_RETRY_BUDGET_DEFAULT`) and prefers highest-quality
healthy models while budget remains. On the last attempt (`<= 1`) it failovers
to the lowest rolling-p95 healthy model so GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic still has a fast final try.

```dotenv
NEXUS_RETRY_BUDGET_DEFAULT=3
```

`NEXUS_RETRY_BUDGET_DEFAULT` is the default remaining retry count when metadata
omits `retry_remaining` (integer >= 0; default `3`).

See
[docs/guides/RETRY_BUDGET_AWARE_FAILOVER_GUIDE.md](docs/guides/RETRY_BUDGET_AWARE_FAILOVER_GUIDE.md).


## Cache-Hit-Sticky-Warm-Pool Routing

The `cache-hit-sticky-warm-pool` strategy pins identical long prompt prefixes
onto one domain-eligible model so provider prompt caches stay warm for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. Short prefixes fall back to
`session_id`. Unhealthy sticky primaries failover along a deterministic ring.

```dotenv
NEXUS_CACHE_HIT_STICKY_MIN_CHARS=64
```

`NEXUS_CACHE_HIT_STICKY_MIN_CHARS` is the minimum prefix length before sticky
hashing activates (integer >= 1; default `64`).

See
[docs/guides/CACHE_HIT_STICKY_WARM_POOL_GUIDE.md](docs/guides/CACHE_HIT_STICKY_WARM_POOL_GUIDE.md).

## Embedding-Cache-Key-Namespace Routing

The `embedding-cache-key-namespace` strategy pins sticky/cache routing under a
tenant-isolated namespace so embedding-cache keys do not collide across tenants
for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

```dotenv
NEXUS_EMBEDDING_CACHE_NAMESPACE_PREFIX=embed
```

`NEXUS_EMBEDDING_CACHE_NAMESPACE_PREFIX` is prepended to the tenant/session
scope before sticky hashing (non-empty string; default `embed`).

See
[docs/guides/EMBEDDING_CACHE_KEY_NAMESPACE_GUIDE.md](docs/guides/EMBEDDING_CACHE_KEY_NAMESPACE_GUIDE.md).


## Circuit-Breaker-Half-Open-Probe Routing

The `circuit-breaker-half-open-probe` strategy prefers healthy closed providers
and allows only limited concurrent probes into half-open/recovering providers
for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

```dotenv
NEXUS_CIRCUIT_HALF_OPEN_PROBE_BUDGET=2
```

`NEXUS_CIRCUIT_HALF_OPEN_PROBE_BUDGET` is the maximum concurrent in-flight
attempts across half-open providers before probes are deferred (integer >= 1;
default `2`).

See
[docs/guides/CIRCUIT_BREAKER_HALF_OPEN_PROBE_GUIDE.md](docs/guides/CIRCUIT_BREAKER_HALF_OPEN_PROBE_GUIDE.md).


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

## Adaptive-Exploration Routing

The `adaptive-exploration` strategy is a decaying epsilon-greedy bandit over the
model catalog. It starts at `NEXUS_ADAPTIVE_EXPLORATION_BASE` explore probability
while `SuccessStats` are empty, then decays toward
`NEXUS_ADAPTIVE_EXPLORATION_MIN` as total provider successes grow:

`epsilon = min + (base - min) / (1 + total_successes)`.

Explore/exploit bucketing matches canary/A/B/`epsilon-greedy`, so a given
request stays replayable while early traffic still samples GPT-5.5, Claude Sonnet
4.6, Gemini 3.x, and Kimi K2 more broadly than a fixed epsilon.

```dotenv
NEXUS_ADAPTIVE_EXPLORATION_BASE=0.2
NEXUS_ADAPTIVE_EXPLORATION_MIN=0.02
```

Both values are explore probabilities within `[0.0, 1.0]`; `min` must be `<=`
`base`. Defaults are `0.2` → `0.02`. See
[docs/guides/ADAPTIVE_EXPLORATION_GUIDE.md](docs/guides/ADAPTIVE_EXPLORATION_GUIDE.md).

## Sticky-Region-Failover Routing

The `sticky-region-failover` strategy combines geo-region preference with
session stickiness and ordered failover. It walks an ordered region list (request
`region` first, then `NEXUS_STICKY_REGION_FAILOVER_PREFERENCES`), selects the
first region with at least one healthy domain-eligible model, and pins
`session_id` to one model in that pool via consistent hashing. When the
preferred region has no healthy providers it advances to the next region while
keeping sticky affinity inside the active pool.

```dotenv
NEXUS_STICKY_REGION_FAILOVER_PREFERENCES=["eu","us","cn","global"]
```

`NEXUS_STICKY_REGION_FAILOVER_PREFERENCES` is the ordered failover list used
when a request omits `region`. Request `region` is always tried first.

See
[docs/guides/STICKY_REGION_FAILOVER_GUIDE.md](docs/guides/STICKY_REGION_FAILOVER_GUIDE.md).

## Sticky-Tenant-Hash Routing

The `sticky-tenant-hash` strategy pins each tenant to a primary model using
consistent hashing on `metadata.tenant_id` (then `metadata.user_id`,
`metadata.sticky_key`, top-level `user_id`, and finally `session_id`). Unlike
`sticky-session`, which hashes only `session_id` for multi-turn conversational
affinity, this strategy keeps GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
traffic stable per customer across sessions. When the sticky primary provider is
unhealthy the strategy walks a deterministic ring to the next healthy candidate.

No additional `NEXUS_*` environment variables are required. Select the strategy
via `NEXUS_DEFAULT_STRATEGY=sticky-tenant-hash` or the `X-Router-Strategy`
header and pass `metadata.tenant_id` on each request.

See
[docs/guides/STICKY_TENANT_HASH_GUIDE.md](docs/guides/STICKY_TENANT_HASH_GUIDE.md).

## Multi-Region-Latency-Hedge Routing

The `multi-region-latency-hedge` strategy keeps GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic on the highest-quality model in the request's
primary region (`request.region`, default `global`). When that model's provider
rolling **p50** exceeds `NEXUS_LATENCY_HEDGE_MS`, and at least one
secondary-region candidate exists, it hedges to the lowest-p50 secondary model.
Otherwise it stays on the primary quality preference. Providers with no
observations yet report `0.0` p50 and keep primary routing.

```dotenv
NEXUS_LATENCY_HEDGE_MS=500.0
```

`NEXUS_LATENCY_HEDGE_MS` is the primary-region provider p50 threshold in
milliseconds that triggers hedging (non-negative; default `500`).

See
[docs/guides/MULTI_REGION_LATENCY_HEDGE_GUIDE.md](docs/guides/MULTI_REGION_LATENCY_HEDGE_GUIDE.md).

## Token-Budget Routing

The `token-budget` strategy maximizes quality subject to a hard token ceiling: it
selects the highest-quality domain-eligible model whose
`min(context_window, request.token_budget)` can hold
`prompt_tokens_estimate + max_tokens`. Useful for long RAG contexts across
GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2 without risking provider
context overflows. When no model fits it falls back to the largest-context
eligible candidate. Requires no additional env vars; set `token_budget` on the
request (default `4096`). See
[docs/guides/TOKEN_BUDGET_GUIDE.md](docs/guides/TOKEN_BUDGET_GUIDE.md).

## SLO-Aware Routing

The `slo-aware` strategy maximizes quality subject to a rolling availability SLO:
it selects the highest-quality domain-eligible model whose provider success rate
meets `NEXUS_AVAILABILITY_SLO` (default `0.99`). Useful when soft degradation
would otherwise keep routing to GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi
K2 providers that are burning error budget. Providers with no observations yet
are treated as healthy; when nothing meets the SLO it falls back to the highest
success-rate eligible model. See
[docs/guides/SLO_AWARE_GUIDE.md](docs/guides/SLO_AWARE_GUIDE.md).

```dotenv
NEXUS_AVAILABILITY_SLO=0.99
```

## Adaptive-Timeout Routing

The `adaptive-timeout` strategy combines the latency-budget and SLO-aware signal
stores: it derives a timeout budget from the request latency requirement,
provider rolling p95 latency, and recent success/error signals. Realtime
requests use `NEXUS_LATENCY_SLA_MS` as the base budget, batch requests receive a
wider budget, and provider failures inflate the effective latency. Useful when
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic should prefer faster
models during provider latency spikes but keep higher-quality slower models when
the budget is comfortable. See
[docs/guides/ADAPTIVE_TIMEOUT_GUIDE.md](docs/guides/ADAPTIVE_TIMEOUT_GUIDE.md).

```dotenv
NEXUS_DEFAULT_STRATEGY=adaptive-timeout
NEXUS_LATENCY_SLA_MS=750
```

## Semantic-Cache Routing

The `semantic-cache` strategy is Portkey/LiteLLM-style cache-aware routing: when
`request.metadata.cache_hit` is truthy it prefers the cheapest domain-eligible
realtime model among GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2; on a
miss it falls through to `cost-optimal` under `NEXUS_QUALITY_FLOOR`. Useful when
an upstream semantic cache already resolved the answer and frontier spend would
be wasted. See
[docs/guides/SEMANTIC_CACHE_STRATEGY_GUIDE.md](docs/guides/SEMANTIC_CACHE_STRATEGY_GUIDE.md).

## Failover-Priority Routing

The `failover-priority` strategy is LiteLLM-style ordered failover: it walks
`NEXUS_FAILOVER_PRIORITY` (default GPT-5.5 → Claude Sonnet 4.6 → Gemini 3.x →
Kimi K2) and selects the first model whose provider circuit is closed. Unhealthy
providers are skipped; when every preference is unhealthy it still routes to the
first listed catalog model. The fallback chain preserves the remaining priority
order. See
[docs/guides/FAILOVER_PRIORITY_GUIDE.md](docs/guides/FAILOVER_PRIORITY_GUIDE.md).

```dotenv
NEXUS_FAILOVER_PRIORITY=["gpt-5.5","claude-sonnet-4-6","gemini-3.1-pro-preview","kimi-k2"]
```

## Provider-Health Score Blend Routing
The `provider-health-score-blend` strategy is LiteLLM/Portkey-style
health-aware routing for the default GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 catalog mix. It scores domain-eligible models by blending rolling
provider success rate, inverse normalized provider p95 latency, model
`quality_score`, and inverse normalized estimated request cost. Provider circuit
availability is a hard gate: when any candidate's circuit is closed, open
circuits are excluded from primary scoring; when every circuit is open, Nexus
still returns the best scored model so decide-time remains deterministic.
```dotenv
NEXUS_HEALTH_BLEND_SUCCESS_WEIGHT=0.35
NEXUS_HEALTH_BLEND_LATENCY_WEIGHT=0.25
NEXUS_HEALTH_BLEND_QUALITY_WEIGHT=0.25
NEXUS_HEALTH_BLEND_COST_WEIGHT=0.15
```
Weights are non-negative and normalized to sum to one, so only ratios matter.
All-zero weights fall back to pure quality. See
[docs/guides/PROVIDER_HEALTH_SCORE_BLEND_GUIDE.md](docs/guides/PROVIDER_HEALTH_SCORE_BLEND_GUIDE.md).

## Health/Cost/Latency Routing
The `health-cost-latency` strategy is LiteLLM/Portkey-style ternary routing for
the default GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog mix. It
scores domain-eligible models by blending rolling provider success rate, inverse
normalized estimated request cost, and inverse normalized rolling provider p95
latency — without circuit-breaker gating or a separate quality component.
```dotenv
NEXUS_HCL_HEALTH_WEIGHT=0.4
NEXUS_HCL_COST_WEIGHT=0.3
NEXUS_HCL_LATENCY_WEIGHT=0.3
```
Weights are non-negative and normalized to sum to one, so only ratios matter.
All-zero weights fall back to pure health (success rate). See
[docs/guides/HEALTH_COST_LATENCY_GUIDE.md](docs/guides/HEALTH_COST_LATENCY_GUIDE.md).

## Soft-Rate-Limit Routing
The `soft-rate-limit` strategy is LiteLLM/Portkey-style soft backoff for
providers that recently returned 429 or rate-limit shaped errors. It prefers
healthy domain-eligible providers with fewer recent rate-limit observations,
then breaks ties by quality and estimated request cost. Successful attempts add
non-rate-limit observations, so GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi
K2 traffic can return to a provider after quota pressure cools down. No
additional `NEXUS_*` setting is required; select it with
`NEXUS_DEFAULT_STRATEGY=soft-rate-limit` or per request with
`X-Router-Strategy: soft-rate-limit`. See
[docs/guides/SOFT_RATE_LIMIT_GUIDE.md](docs/guides/SOFT_RATE_LIMIT_GUIDE.md).
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
## Prompt-Prefix-Cache Routing
The `prompt-prefix-cache` strategy gives long reusable system prompts sticky
provider/model affinity for OpenRouter/LiteLLM-style prompt caching. It hashes
the first `NEXUS_PROMPT_PREFIX_CACHE_MIN_CHARS` characters of joined `system`
messages and buckets that prefix across domain-eligible candidates, so requests
sharing a long prefix keep hitting the same GPT-5.5 / Claude Sonnet 4.6 / Gemini
3.x / Kimi K2 provider/model cache. Requests without a sufficiently long system
prompt fall back to `cost-optimal` under `NEXUS_QUALITY_FLOOR`.
NEXUS_PROMPT_PREFIX_CACHE_MIN_CHARS=512
See
[docs/guides/PROMPT_PREFIX_CACHE_STRATEGY_GUIDE.md](docs/guides/PROMPT_PREFIX_CACHE_STRATEGY_GUIDE.md).
## Cost/Latency Pareto Routing
The `cost-latency-pareto` strategy keeps only non-dominated domain-eligible
candidates under estimated request cost and rolling provider p95 latency, then
breaks ties by higher `quality_score` (then lower cost, lower latency, and model
name). Cold providers start at p95 `0.0`, so equal-latency startups collapse to
the cheapest frontier before the quality tie-break. No additional `NEXUS_*`
setting is required; select it with `NEXUS_DEFAULT_STRATEGY=cost-latency-pareto`
or per request with `X-Router-Strategy: cost-latency-pareto`. See
[docs/guides/COST_LATENCY_PARETO_GUIDE.md](docs/guides/COST_LATENCY_PARETO_GUIDE.md).

## Concurrency-Cap Routing
The `concurrency-cap` strategy is a LiteLLM/Portkey-style provider saturation
guard for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. It reads
the same live `InflightStats` counters as `least-busy`, skips providers whose
current in-flight count is at or above `NEXUS_CONCURRENCY_CAP`, and selects the
highest-quality domain-eligible model among the remaining providers. If every
eligible provider is already capped, it degrades deterministically to the
least-loaded eligible provider and records that condition in the rationale.

```dotenv
NEXUS_DEFAULT_STRATEGY=concurrency-cap
NEXUS_CONCURRENCY_CAP=8
```

`NEXUS_CONCURRENCY_CAP` is the maximum live attempts per provider before new
primary traffic is steered elsewhere (minimum `1`). See
[docs/guides/CONCURRENCY_CAP_GUIDE.md](docs/guides/CONCURRENCY_CAP_GUIDE.md).

## Token-Bucket-Burst Routing
The `token-bucket-burst` strategy is LiteLLM/Portkey/OpenRouter-style bursty quota
routing for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. Each
provider keeps a shared token bucket that refills over time. Primary selection
prefers providers with at least one available token, then breaks ties by quality
and estimated request cost. When every bucket is empty, Nexus falls back to the
highest remaining token fraction, then to the cheapest eligible model, and still
consumes one token from the chosen provider so buckets drain gradually instead
of hard-blocking.

```dotenv
NEXUS_DEFAULT_STRATEGY=token-bucket-burst
NEXUS_TOKEN_BUCKET_CAPACITY=10
NEXUS_TOKEN_BUCKET_REFILL_PER_SEC=1.0
```

`NEXUS_TOKEN_BUCKET_CAPACITY` is the per-provider burst ceiling (minimum `1`).
`NEXUS_TOKEN_BUCKET_REFILL_PER_SEC` is the refill rate in tokens per second and
must be positive. See
[docs/guides/TOKEN_BUCKET_BURST_GUIDE.md](docs/guides/TOKEN_BUCKET_BURST_GUIDE.md).

## Model-Tier-Rate-Limit Routing
The `model-tier-rate-limit` strategy is LiteLLM/Portkey-style tier-aware soft RPM
routing for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. It
infers each candidate's frontier/mid/economy tier from its model name, tracks
rolling per-provider request timestamps, and prefers providers still under the
tier-specific RPM ceiling. When every eligible provider is saturated it falls
back to the least-saturated provider, then quality and estimated cost.

```dotenv
NEXUS_DEFAULT_STRATEGY=model-tier-rate-limit
NEXUS_TIER_FRONTIER_RPM=30
NEXUS_TIER_MID_RPM=60
NEXUS_TIER_ECONOMY_RPM=120
```

`NEXUS_TIER_FRONTIER_RPM` defaults to `30`, `NEXUS_TIER_MID_RPM` to `60`, and
`NEXUS_TIER_ECONOMY_RPM` to `120`. Each must be at least `1`. See
[docs/guides/MODEL_TIER_RATE_LIMIT_GUIDE.md](docs/guides/MODEL_TIER_RATE_LIMIT_GUIDE.md).

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
- `adaptive-exploration`
- `geo-region`
- `region-tier-affinity`
- `token-budget`
- `slo-aware`
- `semantic-cache`
- `least-busy`
- `prompt-prefix-cache`
- `concurrency-cap`
- `soft-rate-limit`
- `cost-latency-pareto`
- `token-bucket-burst`
- `model-tier-rate-limit`
- `failover-priority`
- `provider-health-score-blend`
- `health-cost-latency`
- `embedding-cache-key-namespace`
- `circuit-breaker-half-open-probe`
- `ab`

If the header is absent, Nexus uses `NEXUS_DEFAULT_STRATEGY`.

## Built-In Model Catalog

The default catalog lives in `src/router/config.py` and includes:

- OpenAI: `gpt-5.5`, `gpt-4.1-mini`
- Anthropic: `claude-sonnet-4-6`, `claude-haiku-4-5`
- Google: `gemini-3.1-pro-preview`, `gemini-3.5-flash`
- Moonshot: `kimi-k2`

Each model has quality, cost, domain, and realtime-support priors. Replace or extend the catalog when onboarding provider-specific SKUs.

## Canary-Cost-Blend Routing
The `canary-cost-blend` strategy is a **cost-first** counterpart to
NEXUS_CANARY_COST_BLEND_PERCENT=10.0
`NEXUS_CANARY_COST_BLEND_PERCENT` is the percentage of traffic (within
[docs/guides/CANARY_COST_BLEND_GUIDE.md](docs/guides/CANARY_COST_BLEND_GUIDE.md).

## Token-Cost-Anomaly-Shed Routing

The `token-cost-anomaly-shed` strategy tracks a rolling mean cost-per-1k-tokens
via shared `CostAnomalyStats` (populated by the engine after successful
completions). It ranks domain-eligible candidates by quality and, when the top
pick's projected cost/1k exceeds `mean * NEXUS_TOKEN_COST_ANOMALY_RATIO`,
sheds to the cheapest healthy alternative below that cost. When no cheaper
healthy option exists it falls back to pure quality ranking. Cold start (empty
stats) routes to the highest-quality eligible model.

```dotenv
NEXUS_TOKEN_COST_ANOMALY_RATIO=2.0
```

`NEXUS_TOKEN_COST_ANOMALY_RATIO` is the positive multiplier above the rolling
mean cost/1k that triggers shedding (default `2.0`).

See
[docs/guides/TOKEN_COST_ANOMALY_SHED_GUIDE.md](docs/guides/TOKEN_COST_ANOMALY_SHED_GUIDE.md).

## Provider spend telemetry

`NEXUS_PROVIDER_SPEND_SOFT_USD` (default `10.0`) activates lower-spend preference once any provider spend metadata meets the soft threshold.

## Carbon-aware preference

`NEXUS_CARBON_AWARE_MAX_INTENSITY` (default `400.0`).

## Provider-Circuit-Probe Routing

The `provider-circuit-probe` strategy follows the highest-quality eligible
provider's circuit state. It keeps that leader while closed, actively selects
the best healthy alternate while the leader is open, and allows only
`NEXUS_PROVIDER_CIRCUIT_PROBE_BUDGET` probe decisions while the leader is
half-open. Once that budget is exhausted, GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic remains on a healthy alternate.

```dotenv
NEXUS_DEFAULT_STRATEGY=provider-circuit-probe
NEXUS_PROVIDER_CIRCUIT_PROBE_BUDGET=1
```

See
[docs/guides/PROVIDER_CIRCUIT_PROBE_GUIDE.md](docs/guides/PROVIDER_CIRCUIT_PROBE_GUIDE.md).

## Carbon-Latency-Blend Routing

The `carbon-latency-blend` strategy independently weights inverse-normalized
provider-region carbon intensity and rolling provider p95 latency:
`score = w_carbon * normalized_inverse_carbon + w_latency *
normalized_inverse_latency`. Carbon data resolves from
`carbon_intensity:<provider>:<region>`, then provider-only metadata, then a
built-in provider/region map. Unlike `region-carbon-blend`, these two weights do
not have to be complementary. Defaults are `0.5`/`0.5` for GPT-5.5 / Claude
Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

```dotenv
NEXUS_DEFAULT_STRATEGY=carbon-latency-blend
NEXUS_CARBON_LATENCY_CARBON_WEIGHT=0.5
NEXUS_CARBON_LATENCY_LATENCY_WEIGHT=0.5
```

See
[docs/guides/CARBON_LATENCY_BLEND_GUIDE.md](docs/guides/CARBON_LATENCY_BLEND_GUIDE.md).

## Adaptive-Concurrency-Cap Routing

The `adaptive-concurrency-cap` strategy scales each provider's effective
in-flight cap by rolling success rate and inverse p95 latency:
`effective_cap = max(min_cap, int(base_cap * success_rate * min(1, ref_ms /
p95)))`. Providers at or above their adaptive cap are skipped; the healthiest
remaining provider wins primary selection. Defaults target GPT-5.5 / Claude
Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

```dotenv
NEXUS_DEFAULT_STRATEGY=adaptive-concurrency-cap
NEXUS_ADAPTIVE_CONCURRENCY_BASE_CAP=8
NEXUS_ADAPTIVE_CONCURRENCY_MIN_CAP=1
NEXUS_ADAPTIVE_CONCURRENCY_LATENCY_MS=2000.0
```

See
[docs/guides/ADAPTIVE_CONCURRENCY_CAP_GUIDE.md](docs/guides/ADAPTIVE_CONCURRENCY_CAP_GUIDE.md).

## Provider-Token-Fair-Share Routing

The `provider-token-fair-share` strategy tracks estimated prompt tokens per
provider in a rolling 60-second window and selects providers with the most
remaining fair-share headroom under
`NEXUS_PROVIDER_TOKEN_FAIR_SHARE_CEILING` (default `100000`). Ties break with
request-id weighted round-robin. When every provider is over quota, traffic
falls back to the least-used provider. Targets GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic.

```dotenv
NEXUS_DEFAULT_STRATEGY=provider-token-fair-share
NEXUS_PROVIDER_TOKEN_FAIR_SHARE_CEILING=100000
```

See
[docs/guides/PROVIDER_TOKEN_FAIR_SHARE_GUIDE.md](docs/guides/PROVIDER_TOKEN_FAIR_SHARE_GUIDE.md).

## Region-Failover-Hysteresis Routing

The `region-failover-hysteresis` strategy combines ordered region preference,
session stickiness, and hysteresis before returning to a preferred region
after failover. It reuses `NEXUS_STICKY_REGION_FAILOVER_PREFERENCES` and waits
for `NEXUS_REGION_FAILOVER_HYSTERESIS_SUCCESSES` (default `3`) consecutive
preferred-region successes recorded in shared engine state. Targets GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

```dotenv
NEXUS_DEFAULT_STRATEGY=region-failover-hysteresis
NEXUS_REGION_FAILOVER_HYSTERESIS_SUCCESSES=3
NEXUS_STICKY_REGION_FAILOVER_PREFERENCES=eu,us,cn,global
```

See
[docs/guides/REGION_FAILOVER_HYSTERESIS_GUIDE.md](docs/guides/REGION_FAILOVER_HYSTERESIS_GUIDE.md).

## Tenant-Budget-Cascade Routing

The `tenant-budget-cascade` strategy tracks successful completion spend in an
in-memory one-hour window per tenant. Requests stay quality-first while their
projected cost fits `NEXUS_TENANT_BUDGET_CASCADE_SOFT` (default `10.0`). Once
soft headroom is exhausted they shed to the cheapest hard-safe provider. If no
eligible request fits `NEXUS_TENANT_BUDGET_CASCADE_HARD` (default `12.5`), the
strategy fails closed with a tenant-specific rationale. This protects GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 tenant budgets.

```dotenv
NEXUS_DEFAULT_STRATEGY=tenant-budget-cascade
NEXUS_TENANT_BUDGET_CASCADE_SOFT=10.0
NEXUS_TENANT_BUDGET_CASCADE_HARD=12.5
```

The hard threshold must be greater than the soft threshold.

See
[docs/guides/TENANT_BUDGET_CASCADE_GUIDE.md](docs/guides/TENANT_BUDGET_CASCADE_GUIDE.md).

## Provider-Error-Budget-Reset Routing

The `provider-error-budget-reset` strategy tracks attempts and errors in an
independent timed window per provider. It temporarily sheds providers whose
error rate exceeds `NEXUS_PROVIDER_ERROR_BUDGET_RESET_FRACTION` (default
`0.15`) and restores them after `NEXUS_PROVIDER_ERROR_BUDGET_RESET_SECONDS`
(default `60`). This explicit timer distinguishes it from
`provider-error-budget-shed`, which reads cumulative shared `SuccessStats`.
The strategy targets GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

```dotenv
NEXUS_DEFAULT_STRATEGY=provider-error-budget-reset
NEXUS_PROVIDER_ERROR_BUDGET_RESET_FRACTION=0.15
NEXUS_PROVIDER_ERROR_BUDGET_RESET_SECONDS=60.0
```

See
[docs/guides/PROVIDER_ERROR_BUDGET_RESET_GUIDE.md](docs/guides/PROVIDER_ERROR_BUDGET_RESET_GUIDE.md).

## Sticky-Region-Warmup Routing

The `sticky-region-warmup` strategy routes the first
`NEXUS_STICKY_REGION_WARMUP_REQUESTS` requests (default `3`) for each session to
a warmup region, then pins the session to its requested or hash-selected region.
It reuses `NEXUS_STICKY_REGION_FAILOVER_PREFERENCES`; the first preference is the
default warmup region and request metadata may override it with
`metadata.warmup_region`. Durable post-warmup affinity prevents cold-start flaps
across GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 regional pools.

```dotenv
NEXUS_DEFAULT_STRATEGY=sticky-region-warmup
NEXUS_STICKY_REGION_WARMUP_REQUESTS=3
NEXUS_STICKY_REGION_FAILOVER_PREFERENCES=eu,us,cn,global
```

See
[docs/guides/STICKY_REGION_WARMUP_GUIDE.md](docs/guides/STICKY_REGION_WARMUP_GUIDE.md).

## Tenant-Quota-Burst Routing

The `tenant-quota-burst` strategy applies an independent rolling request window
to each tenant. The first `NEXUS_TENANT_QUOTA_BURST_SOFT` requests (default
`60`) stay quality-first. Requests in the burst band are admitted but shed to
the cheapest domain-compatible fallback until
`NEXUS_TENANT_QUOTA_BURST_HARD` (default `75`) is reached. Further requests fail
closed before provider dispatch. `NEXUS_TENANT_QUOTA_BURST_WINDOW_SECONDS`
(default `60`) controls when request slots expire.

```dotenv
NEXUS_DEFAULT_STRATEGY=tenant-quota-burst
NEXUS_TENANT_QUOTA_BURST_SOFT=60
NEXUS_TENANT_QUOTA_BURST_HARD=75
NEXUS_TENANT_QUOTA_BURST_WINDOW_SECONDS=60.0
```

The hard quota must be greater than the soft quota. State is in-memory and
process-local; use a shared rate-limit service when replicas need a global
quota. This strategy supports GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic.

See
[docs/guides/TENANT_QUOTA_BURST_GUIDE.md](docs/guides/TENANT_QUOTA_BURST_GUIDE.md).

## Provider-Tail-Latency-Hedge Routing

The `provider-tail-latency-hedge` strategy normally selects the highest-quality
healthy model. It hedges across providers only when that model's rolling p95
latency exceeds `NEXUS_PROVIDER_TAIL_LATENCY_HEDGE_MS` (default `1500`). The
fastest observed healthy provider alternative then becomes the primary route.
Providers without latency samples cannot attract a hedge based on missing data.

```dotenv
NEXUS_DEFAULT_STRATEGY=provider-tail-latency-hedge
NEXUS_PROVIDER_TAIL_LATENCY_HEDGE_MS=1500.0
```

This differs from `multi-region-latency-hedge`, which uses primary-region p50,
and `adaptive-timeout-hedge`, which derives a relative p95 threshold from the
fastest provider. It supports GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic.

See
[docs/guides/PROVIDER_TAIL_LATENCY_HEDGE_GUIDE.md](docs/guides/PROVIDER_TAIL_LATENCY_HEDGE_GUIDE.md).

## Sticky-Session-Migrate Routing

The `sticky-session-migrate` strategy stores a model pin for each `session_id`
and preserves it while the provider is available and its observed success rate
meets `NEXUS_STICKY_SESSION_MIGRATE_SUCCESS_THRESHOLD` (default `0.9`). When the
pinned provider becomes unavailable or falls below the threshold, the session
migrates to the highest-success healthy provider that meets the threshold. The
new pin remains stable after the original provider recovers, avoiding automatic
failback during an active conversation.

```dotenv
NEXUS_DEFAULT_STRATEGY=sticky-session-migrate
NEXUS_STICKY_SESSION_MIGRATE_SUCCESS_THRESHOLD=0.9
```

The threshold must be between `0.0` and `1.0`. Providers without observations
start at a `1.0` success rate. Session pins are process-local and reset when the
router restarts. This strategy supports GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic.

See
[docs/guides/STICKY_SESSION_MIGRATE_GUIDE.md](docs/guides/STICKY_SESSION_MIGRATE_GUIDE.md).

## Provider-Cold-Start-Bias Routing

The `provider-cold-start-bias` strategy tracks recent provider selections in a
bounded process-local window. While a healthy provider has fewer than
`NEXUS_PROVIDER_COLD_START_TARGET` observations (default `5`), the strategy
prefers the least-observed healthy provider and uses quality and cost as
tie-breakers. Once every healthy provider reaches the target, it returns to
quality-first routing until older observations expire.

```dotenv
NEXUS_DEFAULT_STRATEGY=provider-cold-start-bias
NEXUS_PROVIDER_COLD_START_LOOKBACK=100
NEXUS_PROVIDER_COLD_START_TARGET=5
```

`NEXUS_PROVIDER_COLD_START_LOOKBACK` must be at least `1` and controls how many
provider selections are retained. `NEXUS_PROVIDER_COLD_START_TARGET` must also
be at least `1`. Provider circuit health is checked before exploration, so a
cold unhealthy provider cannot displace a healthy alternate serving GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. See
[docs/guides/PROVIDER_COLD_START_BIAS_GUIDE.md](docs/guides/PROVIDER_COLD_START_BIAS_GUIDE.md).

## Tenant-Fair-Queue Routing

The `tenant-fair-queue` strategy computes equal fair share from recent request
counts across active tenants. A tenant at or below its share receives the
highest-quality healthy route. A tenant above share uses the cheapest healthy
relief lane until the bounded request window rebalances.

```dotenv
NEXUS_DEFAULT_STRATEGY=tenant-fair-queue
NEXUS_TENANT_FAIR_QUEUE_LOOKBACK=100
```

`NEXUS_TENANT_FAIR_QUEUE_LOOKBACK` is the number of process-local tenant
decisions retained (integer >= 1; default `100`). Tenant identity resolves from
`metadata.tenant_id`, metadata user/sticky keys, `user_id`, then `session_id`.
This is request-count fairness across tenants, unlike provider
`queue-depth-fairness` or `provider-quota-fair-share`, and supports GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. See
[docs/guides/TENANT_FAIR_QUEUE_GUIDE.md](docs/guides/TENANT_FAIR_QUEUE_GUIDE.md).

## Sticky-Region-Drain Routing

The `sticky-region-drain` strategy keeps each session on a healthy regional pin
until operators mark that region for drain. It then walks
`NEXUS_STICKY_REGION_FAILOVER_PREFERENCES`, moves the session to the first
healthy non-draining region, and persists the alternate pin without automatic
failback.

```dotenv
NEXUS_DEFAULT_STRATEGY=sticky-region-drain
NEXUS_STICKY_REGION_DRAIN_REGIONS=["us"]
NEXUS_STICKY_REGION_FAILOVER_PREFERENCES=["us","eu","cn","global"]
```

`NEXUS_STICKY_REGION_DRAIN_REGIONS` is a JSON list and defaults to `[]`.
Requests may add live markers through `metadata.draining_regions`. When no
healthy non-draining alternate exists, deterministic emergency fallback remains
available. This Envoy/service-mesh-style drain is distinct from regional warmup,
provider-driven session migration, and failback hysteresis for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. See
[docs/guides/STICKY_REGION_DRAIN_GUIDE.md](docs/guides/STICKY_REGION_DRAIN_GUIDE.md).

## Provider-Canary-Shadow-Split Routing

The `provider-canary-shadow-split` strategy keeps user-visible primary traffic
on `NEXUS_PROVIDER_CANARY_PRIMARY_PROVIDER` when that provider is healthy. A
deterministic tenant/request cohort may also identify the highest-quality
healthy model on another provider for shadow comparison. If the preferred
provider is unavailable, primary routing falls back to the highest-quality
healthy candidate.

```dotenv
NEXUS_DEFAULT_STRATEGY=provider-canary-shadow-split
NEXUS_PROVIDER_CANARY_PRIMARY_PROVIDER=openai
NEXUS_PROVIDER_CANARY_SHADOW_PERCENT=5.0
```

The shadow percentage accepts `0.0` through `100.0`. Tenant identity comes from
request metadata or `user_id`, with `request_id` as the fallback hash key.
Process-local `CanaryShadowSplitStats` tracks primary, shadow-provider, and
provider-pair counts. Nexus returns one primary decision; its rationale and
first fallback expose the shadow candidate to downstream dual-run integrations
for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic. See
[docs/guides/PROVIDER_CANARY_SHADOW_SPLIT_GUIDE.md](docs/guides/PROVIDER_CANARY_SHADOW_SPLIT_GUIDE.md).

## Sticky-Model-Pin-Expire Routing

The `sticky-model-pin-expire` strategy gives each `session_id` a process-local
model pin. The router preserves an unexpired pin while its provider remains
healthy. At the TTL deadline, the pin is removed and the next decision
re-evaluates domain eligibility, circuit health, model quality, and estimated
cost before creating a fresh pin.

```dotenv
NEXUS_DEFAULT_STRATEGY=sticky-model-pin-expire
NEXUS_STICKY_MODEL_PIN_TTL_SECONDS=300.0
```

`NEXUS_STICKY_MODEL_PIN_TTL_SECONDS` must be greater than `0.0`.
`StickyModelPinExpireStats` records per-session expiration counts. An unhealthy
pinned provider triggers an early reselection without counting a TTL expiration.
This is model-affinity lifetime management, distinct from
`sticky-region-drain` operational evacuation and `sticky-session-migrate`
success-threshold migration for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic. See
[docs/guides/STICKY_MODEL_PIN_EXPIRE_GUIDE.md](docs/guides/STICKY_MODEL_PIN_EXPIRE_GUIDE.md).

## Tenant-Priority-Lanes Routing

The `tenant-priority-lanes` strategy assigns tenant ids to high, normal, or low
service lanes. During provider-health pressure or when a lane reaches its recent
soft quota, high tenants use the fastest observed healthy provider, normal
tenants remain quality-first, and low tenants use a cost-efficient relief route.

```dotenv
NEXUS_DEFAULT_STRATEGY=tenant-priority-lanes
NEXUS_TENANT_PRIORITY_HIGH_TENANTS=["interactive-premium"]
NEXUS_TENANT_PRIORITY_LOW_TENANTS=["offline-batch"]
NEXUS_TENANT_PRIORITY_LANE_LOOKBACK=100
NEXUS_TENANT_PRIORITY_HIGH_QUOTA=100
NEXUS_TENANT_PRIORITY_NORMAL_QUOTA=60
NEXUS_TENANT_PRIORITY_LOW_QUOTA=30
```

All quotas and the lookback are integers of at least `1`. The tenant lists must
not overlap. Requests may override their configured mapping with
`metadata.priority_lane` set to `high`, `normal`, or `low`; invalid values fall
back to the mapping. The bounded lane counters are process-local, and the lanes
change routing preference rather than preempting in-flight work. This provides
capacity-aware differentiation for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic. See
[docs/guides/TENANT_PRIORITY_LANES_GUIDE.md](docs/guides/TENANT_PRIORITY_LANES_GUIDE.md).

## Request-Class-QoS Routing

The `request-class-qos` strategy reads `metadata.request_class` or
`metadata.qos_class` and applies a class-specific healthy-provider policy:

```dotenv
NEXUS_DEFAULT_STRATEGY=request-class-qos
```

Supported classes are `interactive` (default), `batch`, and `bulk`. Interactive
selects the lowest observed healthy latency with quality tie-breaks, batch keeps
quality-first mid-cost preference, and bulk minimizes estimated request cost.
Unknown values fall back to interactive. When every provider circuit is open the
strategy retains an emergency eligible route for GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic. See
[docs/guides/REQUEST_CLASS_QOS_GUIDE.md](docs/guides/REQUEST_CLASS_QOS_GUIDE.md).
