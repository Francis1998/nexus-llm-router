# Carbon-Latency-Blend Routing Guide

Use `carbon-latency-blend` when GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic needs an explicit two-weight trade-off between provider-region
carbon intensity and rolling provider latency.

![Carbon latency blend demo](../../assets/carbon-latency-blend.gif)

## When to use it

- Carbon intensity and response latency should both influence every decision.
- Provider-region intensity data is available in request metadata.
- Carbon and latency weights need to vary independently rather than sum to one.

## How it works

1. Filter the catalog to domain-eligible models.
2. Resolve carbon intensity from
   `carbon_intensity:<provider>:<region>`, then
   `carbon_intensity:<provider>`, then the built-in provider/region map.
3. Read rolling provider p95 latency from shared `LatencyStats`.
4. Inverse min-max normalize carbon intensity and latency so lower values score
   higher.
5. Compute
   `score = w_carbon * normalized_inverse_carbon + w_latency * normalized_inverse_latency`.
6. Select the highest score; quality and estimated cost break ties.

`carbon-latency-blend` differs from `carbon-aware-preference`, which applies an
intensity cap, and `region-carbon-blend`, which exposes one complementary carbon
weight. These two weights are independent:

```bash
export NEXUS_DEFAULT_STRATEGY=carbon-latency-blend
export NEXUS_CARBON_LATENCY_CARBON_WEIGHT=0.5
export NEXUS_CARBON_LATENCY_LATENCY_WEIGHT=0.5
```

Or per request:

```http
X-Router-Strategy: carbon-latency-blend
```

Example provider-region telemetry:

```json
{
  "region": "eu",
  "carbon_intensity:anthropic:eu": "180",
  "carbon_intensity:openai:eu": "260"
}
```
