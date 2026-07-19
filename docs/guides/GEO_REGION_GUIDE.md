# Geo-Region Routing Guide

Use the `geo-region` strategy when requests carry a data-residency or latency affinity region and you want the highest-quality domain-eligible model that serves that region.

## When to use it

- EU or CN residency requirements must keep traffic on region-capable providers.
- You still want quality-first selection among GPT-5.5, Claude Sonnet 4.6, Gemini 2.5, and Kimi K2 within the allowed region set.
- Missing or unknown regions should fail soft (fall back to best eligible) rather than reject the request.

## How it works

1. Filter to domain-eligible catalog candidates.
2. Keep candidates whose `supported_regions` include the request `region` (default `global` when omitted).
3. Pick the highest `quality_score` among matches (ties break toward lower estimated cost).
4. If nothing matches the region, fall back to the highest-quality domain-eligible model.

Catalog priors (examples): OpenAI US/global, Anthropic/Google US+EU+global, Moonshot (Kimi K2) CN/global.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=geo-region
```

Or per request:

```http
X-Router-Strategy: geo-region
```

Pass `region` on the internal `RouterRequest` (for example `eu`, `us`, `cn`).

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
