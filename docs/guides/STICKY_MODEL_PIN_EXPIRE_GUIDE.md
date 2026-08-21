# Sticky-Model-Pin-Expire Routing Guide

Use `sticky-model-pin-expire` when a session needs temporary model affinity but
must periodically re-evaluate provider health and model quality. Each GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pin expires after a configurable TTL,
then the next request establishes a fresh model pin.

## When to use it

- Conversation turns benefit from stable provider prompt-cache affinity.
- Permanent stickiness would prevent sessions from adopting a healthier model.
- Model affinity needs a clear, auditable lifetime.

## How it works

1. Resolve a process-local model pin by `session_id`.
2. Keep the pin while its monotonic TTL remains and its provider is healthy.
3. Remove the pin exactly at `NEXUS_STICKY_MODEL_PIN_TTL_SECONDS`.
4. Re-evaluate domain eligibility, provider circuit health, quality, and cost.
5. Pin the selected quality leader for a fresh TTL.
6. If the pinned provider becomes unavailable before expiry, reselect early so
   stickiness never defeats availability.

`StickyModelPinExpireStats` exposes expiration counts per session. State is
process-local and resets when the router restarts. This differs from
`sticky-region-drain`, which evacuates operational regions and keeps the new
region pin, and from `sticky-session-migrate`, which uses rolling provider
success rather than a fixed model-pin lifetime.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=sticky-model-pin-expire
export NEXUS_STICKY_MODEL_PIN_TTL_SECONDS=300.0
```

Or select it per request:

```http
X-Router-Strategy: sticky-model-pin-expire
```
