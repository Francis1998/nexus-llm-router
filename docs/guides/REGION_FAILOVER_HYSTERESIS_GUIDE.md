# Region-Failover-Hysteresis Routing Guide

Use `region-failover-hysteresis` when region preference should survive brief
recovery blips without flapping back to a preferred region too early for
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

![Region failover hysteresis demo](../../assets/region-failover-hysteresis.gif)

## When to use it

- Geo-residency routing must failover when a preferred region is unhealthy.
- Returning to the preferred region should wait for sustained recovery.
- Session stickiness inside the active region pool is still required.

## How it works

1. Walk the ordered region preference list (request `region` first, then
   `NEXUS_STICKY_REGION_FAILOVER_PREFERENCES`).
2. Pin `session_id` to a model inside the first healthy region pool.
3. When the preferred region is unhealthy, failover and record the session's
   active failover region.
4. Do not return to the preferred region until it accumulates
   `NEXUS_REGION_FAILOVER_HYSTERESIS_SUCCESSES` consecutive successes.
5. The engine records region successes/failures into shared hysteresis stats.

Unlike `sticky-region-failover`, this strategy avoids flapping back as soon as
the preferred region briefly becomes healthy again.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=region-failover-hysteresis
export NEXUS_REGION_FAILOVER_HYSTERESIS_SUCCESSES=3
export NEXUS_STICKY_REGION_FAILOVER_PREFERENCES=eu,us,cn,global
```

Or per request:

```http
X-Router-Strategy: region-failover-hysteresis
```
