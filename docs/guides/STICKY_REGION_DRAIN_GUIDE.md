# Sticky-Region-Drain Routing Guide

Use `sticky-region-drain` during regional maintenance or capacity evacuation.
Healthy sessions retain their regional pin, while sessions pinned to a marked
draining region migrate to a healthy alternate and remain there for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

## When to use it

- Operators need to empty a region before maintenance or shutdown.
- Existing sessions should move without waiting for health failures.
- The alternate pin should remain stable after the drain marker is removed.

## How it works

1. Pin each session to its existing pin, requested region, or first configured
   preference.
2. Combine `NEXUS_STICKY_REGION_DRAIN_REGIONS` with optional per-request
   `metadata.draining_regions`.
3. Keep a healthy pin when its region is not draining.
4. If the pin is draining, walk `NEXUS_STICKY_REGION_FAILOVER_PREFERENCES` and
   choose the first healthy non-draining regional pool.
5. Persist the alternate pin, preventing automatic failback and region flapping.
6. If no healthy non-draining alternate exists, preserve deterministic emergency
   fallback behavior so the request can still be handled.

This differs from `sticky-region-warmup`, which stages new sessions;
`sticky-session-migrate`, which reacts to provider health/success; and
`region-failover-hysteresis`, which gates failback after an outage.

## Quick start

Pydantic list settings use JSON syntax:

```bash
export NEXUS_DEFAULT_STRATEGY=sticky-region-drain
export NEXUS_STICKY_REGION_DRAIN_REGIONS='["us"]'
export NEXUS_STICKY_REGION_FAILOVER_PREFERENCES='["us","eu","cn","global"]'
```

Or select it per request and supply a live drain marker:

```http
X-Router-Strategy: sticky-region-drain
```

```json
{"metadata": {"draining_regions": ["us"]}}
```
