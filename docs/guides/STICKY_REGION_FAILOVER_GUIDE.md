# Sticky-Region-Failover Routing Guide

Use the `sticky-region-failover` strategy when geo-residency routing must keep
**session stickiness** inside a preferred region while still recovering when that
region's providers are unhealthy.

## When to use it

- Multi-turn chats need stable model affinity for prompt-cache hits.
- Compliance or latency requires EU / US / CN / global region preference.
- A preferred region can become unhealthy and traffic must failover in order.
- You want sticky sessions without giving up ordered region failover.

## How it works

1. Build an ordered region list: request `region` first, then
   `NEXUS_STICKY_REGION_FAILOVER_PREFERENCES` (default `eu,us,cn,global`).
2. Filter domain-eligible catalog candidates.
3. Walk the region list and pick the first region with at least one **healthy**
   provider offering an eligible model.
4. Consistent-hash `session_id` onto one model in that region pool (same pin as
   `sticky-session`, but scoped to the active region).
5. When no region has healthy providers, fall back to sticky selection across
   all eligible models.

Catalog priors cover GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 across
`us`, `eu`, `cn`, and `global` regions.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=sticky-region-failover
```

Per-request region affinity:

```http
X-Router-Strategy: sticky-region-failover
X-Router-Region: eu
```

Optional failover ordering when requests omit `region`:

```dotenv
NEXUS_STICKY_REGION_FAILOVER_PREFERENCES=["eu","us","cn","global"]
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
