# Region-Latency-P99-Shed Routing Guide

![Region latency p99 shed flow](../../assets/region-latency-p99-shed.gif)

Use the `region-latency-p99-shed` strategy when regional providers with hot
tail latency should be shed before quality ranking concentrates traffic on
them.

## When to use it

- GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic needs
  region affinity plus a soft p99 tail-latency gate.
- You already record provider latency in shared `LatencyStats`.
- Operators want a threshold
  (`NEXUS_REGION_LATENCY_P99_MS`, default `3000`) that is stricter on the
  tail than p95 SLO shedding alone.

## How it works

1. Filter to domain-eligible catalog candidates.
2. Prefer candidates whose `supported_regions` include the request region
   (default `global`); if none match, keep the full eligible set.
3. Compute provider p99 via `LatencyStats.p99(provider)`. Providers with no
   observations are treated as 0ms for cold starts.
4. Among providers under the p99 threshold, select the highest-quality model.
5. If every regional provider is over threshold, select the lowest p99, then
   highest quality.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=region-latency-p99-shed
export NEXUS_REGION_LATENCY_P99_MS=3000
```

Or per request:

```http
X-Router-Strategy: region-latency-p99-shed
```

Optional request region:

```json
{ "region": "eu" }
```

## Tuning notes

- Lower the threshold (for example `1500`) when regional SLOs are tight.
- Raise the threshold (for example `5000`) when transient tail spikes are
  common and preserving regional residency matters more.
- Pair with `latency-slo-shed` for p95 shedding, or `multi-region-latency-hedge`
  when secondary-region hedging is preferred.

## Suggested repo metadata

- **Description:** Multi-provider LLM router with pluggable strategies, safety
  guardrails, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog.
- **Topics:** `llm-router`, `litellm`, `openai`, `anthropic`, `gemini`, `routing`, `python`
