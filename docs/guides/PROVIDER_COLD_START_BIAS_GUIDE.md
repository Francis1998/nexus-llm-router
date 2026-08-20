# Provider-Cold-Start-Bias Routing Guide

Use `provider-cold-start-bias` when healthy providers with sparse recent traffic
need deliberate exploration before quality-first routing settles on a leader.
This closes the observation gaps common in LiteLLM/Portkey-style multi-provider
fleets serving GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## When to use it

- A newly enabled provider needs enough live selections to produce useful
  latency, reliability, or cost telemetry.
- Quiet providers age out of the recent sample window and should be revisited.
- Exploration must never prefer an unhealthy provider over a healthy alternate.

## How it works

1. Filter the catalog to domain-eligible models and healthy providers.
2. Count provider selections in the last `NEXUS_PROVIDER_COLD_START_LOOKBACK`
   routing decisions.
3. While any healthy provider has fewer than
   `NEXUS_PROVIDER_COLD_START_TARGET` observations, select the least-observed
   provider; quality and cost break ties.
4. Record the selected provider in the bounded window.
5. Once every healthy provider reaches the target, return to quality-first
   routing until an observation gap reopens.

If every eligible provider is unhealthy, the strategy preserves Nexus's
emergency fallback behavior and routes deterministically from the eligible pool.
Observation state is local to each router process.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-cold-start-bias
export NEXUS_PROVIDER_COLD_START_LOOKBACK=100
export NEXUS_PROVIDER_COLD_START_TARGET=5
```

Or select it per request:

```http
X-Router-Strategy: provider-cold-start-bias
```
