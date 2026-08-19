# Provider-Tail-Latency-Hedge Routing Guide

Use `provider-tail-latency-hedge` when GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic should preserve quality while typical latency is
healthy but escape a provider whose slow tail crosses an absolute limit.

## When to use it

- Median latency looks healthy while a meaningful tail of requests is slow.
- Hedges should cross providers rather than require a secondary region.
- A fixed p95 objective is more useful than a ratio relative to the fastest
  current provider.

## How it works

1. Filter to domain-compatible models and healthy providers.
2. Select the highest-quality model as the normal primary.
3. Read its provider's rolling p95 from shared `LatencyStats`.
4. If p95 exceeds `NEXUS_PROVIDER_TAIL_LATENCY_HEDGE_MS`, select the observed
   healthy provider alternative with the lowest p95.
5. Otherwise retain the quality leader. Missing latency samples never trigger
   or attract a hedge.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-tail-latency-hedge
export NEXUS_PROVIDER_TAIL_LATENCY_HEDGE_MS=1500.0
```

Or per request:

```http
X-Router-Strategy: provider-tail-latency-hedge
```

## How it differs

- `multi-region-latency-hedge` uses an absolute p50 threshold and requires a
  secondary region.
- `adaptive-timeout-hedge` compares provider p95 values using a relative ratio.
- `latency-slo-shed` filters the whole candidate pool under an SLO; this
  strategy specifically evaluates the quality leader and picks the fastest
  observed cross-provider hedge target only after its tail breaches.

This follows the tail-tolerance principle behind Google hedged requests while
using AWS-style live latency observations to choose the alternate provider.
