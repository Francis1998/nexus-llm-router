# Provider-Success-Floor Routing Guide

Use `provider-success-floor` to skip providers whose recent success rate falls
below a configurable floor while still emergency-retaining a route when every
provider is degraded for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## When to use it

- Multi-provider fleets where soft reliability should gate selection before
  circuits fully open.
- Gateways that already record rolling success/failure telemetry.
- Workloads that prefer quality among providers that still clear a success floor.

## How it works

1. Filter domain-eligible candidates through provider circuit health.
2. Consult shared `SuccessStats` success rates.
3. Keep providers at or above `NEXUS_PROVIDER_SUCCESS_FLOOR` (default `0.85`).
4. Among remaining providers, pick highest quality.
5. If every candidate is below the floor, emergency-retain the highest-success
   provider so the request still decides.
6. Providers with no observations are treated as fully healthy (`1.0`).

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-success-floor
export NEXUS_PROVIDER_SUCCESS_FLOOR=0.85
```

Or select it per request:

```http
X-Router-Strategy: provider-success-floor
```
