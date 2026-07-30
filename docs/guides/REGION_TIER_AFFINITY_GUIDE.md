# Region-Tier-Affinity Routing Guide

Use the `region-tier-affinity` strategy when requests carry a geo/residency
affinity **and** you want capability tier to track prompt complexity —
preferring models that satisfy both signals before falling back, OpenRouter /
LiteLLM style.

## When to use it

- EU or CN residency requirements must keep traffic on region-capable providers
  while still matching frontier / mid / economy to task difficulty.
- You want GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2 catalog priors
  selected by **region + tier** together, not quality alone.
- Missing regions or missing tier SKUs should fail soft (tier → region →
  quality) rather than reject the request.

## How it works

1. Filter to domain-eligible catalog candidates.
2. Resolve the request region (`region`, default `global` when omitted).
3. Map `TaskSignals.complexity_score` to a target tier via `infer_target_tier`:
   - **Frontier** when complexity ≥ `0.7`
   - **Mid** when complexity ≥ `0.35`
   - **Economy** otherwise
4. Classify each candidate with `infer_model_tier` (same heuristics as
   `model-tier-rate-limit`).
5. Prefer pools in order, picking highest `quality_score` inside each pool
   (ties break toward lower estimated cost):
   1. region **and** tier match
   2. tier match only
   3. region match only
   4. highest-quality domain-eligible (quality fallback)

Catalog priors (examples): OpenAI US/global, Anthropic/Google US+EU+global,
Moonshot (Kimi K2) CN/global. No extra `NEXUS_*` knobs are required.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=region-tier-affinity
```

Or per request:

```http
X-Router-Strategy: region-tier-affinity
```

Pass `region` on the internal `RouterRequest` (for example `eu`, `us`, `cn`).
Complexity comes from the observe-phase classifier signals.

## Demo

See the offline routing walkthrough in
[`assets/region-tier-affinity.gif`](../../assets/region-tier-affinity.gif)
(or the shared [`assets/demo.gif`](../../assets/demo.gif)).
