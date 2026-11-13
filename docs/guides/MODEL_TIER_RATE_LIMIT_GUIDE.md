# Model-Tier-Rate-Limit Routing Guide

Use the `model-tier-rate-limit` strategy when provider gateways enforce different
soft RPM ceilings for frontier, mid-tier, and economy SKUs and you want Nexus to
steer GPT-6, Claude Opus 5.5, Gemini 4, and Kimi K3.1 traffic away from
saturated providers before hard rate limits or circuit breakers trip.

## When to use it

- LiteLLM, Portkey, OpenRouter, or native provider traffic mixes frontier and
  economy models and each tier has a different practical RPM budget.
- You want frontier models to back off earlier than economy SKUs on the same
  provider.
- You prefer soft saturation signals over hard blocking: saturated providers stay
  in the fallback chain and can recover as timestamps age out of the rolling
  window.

## How it works

1. Filter to domain-eligible catalog candidates.
2. Infer each candidate's tier from its model name, checking economy markers
   first, then mid, then frontier (names with no marker are mid):
   - **Economy**: `-mini`, `nano`, `flash`, `-luna`, `haiku-light`, `-lite`, etc.
   - **Mid**: `gemini-4-hydrogen` (Gemini 4's fast workhorse), `gpt-6.1-terra`,
     `gpt-4.1`, `claude-haiku`, `gemini-2`, etc.
   - **Frontier**: `gpt-6-astra`, `gpt-6(.1)-sol`, `gpt-5`, `claude-fable`,
     `claude-opus`, `claude-sonnet-5` / `-4`, `gemini-4` (Carbon, Argon),
     `gemini-3`, `kimi-k3`, `kimi-k2`, `o3`, etc.
3. Read each provider's rolling request timestamps from shared `TierRequestStats`.
4. Prefer candidates whose provider is under the tier RPM limit
   (`NEXUS_TIER_FRONTIER_RPM`, `NEXUS_TIER_MID_RPM`, `NEXUS_TIER_ECONOMY_RPM`).
5. When every eligible provider is saturated, fall back to the least-saturated
   provider, then break ties by quality and estimated cost.

Each routing decision records one timestamp against the selected provider so the
window reflects live steering pressure.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=model-tier-rate-limit
export NEXUS_TIER_FRONTIER_RPM=30
export NEXUS_TIER_MID_RPM=60
export NEXUS_TIER_ECONOMY_RPM=120
```

Or per request:

```http
X-Router-Strategy: model-tier-rate-limit
```

`NEXUS_TIER_FRONTIER_RPM` defaults to `30`, `NEXUS_TIER_MID_RPM` to `60`, and
`NEXUS_TIER_ECONOMY_RPM` to `120`. Each must be at least `1`.
