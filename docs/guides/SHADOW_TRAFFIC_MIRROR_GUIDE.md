# Shadow-Traffic-Mirror Routing Guide

Use the `shadow-traffic-mirror` strategy when LiteLLM/OpenRouter-style traffic
needs **cost-optimal primary routing** plus a small deterministic slice whose
audit rationale names a shadow mirror model for dual-run telemetry.

## When to use it

- You want the cheapest model above the quality floor, but need to compare a
  second provider on a small traffic slice without returning two primaries.
- GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 gateways need shadow
  mirrors for latency/quality dashboards while production still serves one model.
- You prefer rationale-level telemetry hooks over changing the decide response.

## How it works

1. Filter domain-eligible candidates that meet `NEXUS_QUALITY_FLOOR` (same
   feasibility rules as `cost-optimal`).
2. Pick the **primary** as the lowest estimated-cost feasible model.
3. Compute a **shadow mirror**: the highest-quality feasible model from a
   different provider than the primary.
4. Hash `request_id` deterministically; when the bucket falls under
   `NEXUS_SHADOW_TRAFFIC_PERCENT` (default `5`), annotate the rationale with the
   shadow mirror for dual-run telemetry. The decide phase still returns only the
   primary candidate.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=shadow-traffic-mirror
export NEXUS_SHADOW_TRAFFIC_PERCENT=5
export NEXUS_QUALITY_FLOOR=0.72
```

Or per request:

```http
X-Router-Strategy: shadow-traffic-mirror
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
