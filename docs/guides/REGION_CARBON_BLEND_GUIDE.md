# Region-Carbon-Blend Routing Guide

Use `region-carbon-blend` when GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic should balance lower carbon intensity against rolling provider
latency instead of optimizing either axis alone.

![Demo](../../assets/region-carbon-blend.gif)

## When to use it

- Sustainability goals prefer greener providers for a request region.
- Latency still matters and pure carbon-first routing can pick a slow backend.
- You want one knob (`NEXUS_REGION_CARBON_BLEND_WEIGHT`) to trade the two scores.

## How it works

1. Filter the catalog to domain-eligible models.
2. Resolve carbon intensity from `carbon_intensity:<provider>` metadata, else a
   regional heuristic (`eu`/`us`/`cn`/`global`).
3. Read rolling provider p95 from shared `LatencyStats`.
4. Inverse min-max normalize intensity and latency so lower is better.
5. Score each candidate as
   `carbon_weight * carbon_score + (1 - carbon_weight) * latency_score`.
6. Select the highest blended score; quality and cost break ties.

`NEXUS_REGION_CARBON_BLEND_WEIGHT` defaults to `0.5` (`0` = latency only,
`1` = carbon only). Distinct from `carbon-aware-preference`, which caps
intensity without blending latency.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=region-carbon-blend
export NEXUS_REGION_CARBON_BLEND_WEIGHT=0.5
```

Or per request:

```http
X-Router-Strategy: region-carbon-blend
```

Optional intensity metadata:

```json
{
  "metadata": {
    "region": "eu",
    "carbon_intensity:anthropic": "200",
    "carbon_intensity:openai": "380"
  },
  "messages": [{"role": "user", "content": "Summarize this report."}]
}
```
