# Retry-After-Respect Routing Guide

Use `retry-after-respect` when GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 providers return HTTP 429 with `Retry-After` and you want the router to
honor that wait instead of immediately reselecting the same backend.

![Demo](../../assets/retry-after-respect.gif)

## When to use it

- Providers emit `Retry-After` (or equivalent) during quota pressure.
- Immediate retries on the same provider waste latency and error budget.
- Healthy alternate providers should absorb traffic until the wait expires.

## How it works

1. Filter the catalog to domain-eligible models.
2. Skip providers still inside a Retry-After cooldown window.
3. Prefer the highest-quality healthy ready provider.
4. When every healthy provider is cooling, fall back to the next healthy
   provider with the soonest remaining wait.
5. The engine records cooldowns on rate-limit failures (header/`retry_after`
   attribute when present, else `NEXUS_RETRY_AFTER_DEFAULT_SECONDS`) and clears
   them on success.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=retry-after-respect
export NEXUS_RETRY_AFTER_DEFAULT_SECONDS=30
```

Or per request:

```http
X-Router-Strategy: retry-after-respect
```

## Tuning notes

- Raise the default when upstreams omit `Retry-After` but need longer cool-downs.
- Lower it for faster recovery in soft-throttle environments.
- Cooldowns are local to each router process unless shared externally.
