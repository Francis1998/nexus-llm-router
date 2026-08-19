# Sticky-Session-Migrate Routing Guide

Use `sticky-session-migrate` when a multi-turn session should keep stable model
affinity but must move away from a degraded provider for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

## When to use it

- Conversation continuity and provider prompt-cache affinity matter.
- A sticky target should not trap a session on an unavailable or error-prone
  provider.
- Recovery should not immediately fail the active session back to its original
  target.

## How it works

1. Hash `session_id` onto a deterministic domain-compatible model for the
   initial pin.
2. Reuse that stored pin while its provider circuit is available and shared
   `SuccessStats` meet `NEXUS_STICKY_SESSION_MIGRATE_SUCCESS_THRESHOLD`.
3. If the provider becomes unavailable or drops below the threshold, rank
   provider alternatives by success rate, quality, and cost.
4. Move the session to the best healthy target that meets the threshold.
5. Persist the new pin after the old provider recovers. Another migration
   occurs only if the current pin later degrades.

When no alternative meets the threshold, the strategy preserves the existing
pin and returns a health-ordered fallback chain rather than oscillating between
unhealthy targets. Pins and migration counts are process-local.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=sticky-session-migrate
export NEXUS_STICKY_SESSION_MIGRATE_SUCCESS_THRESHOLD=0.9
```

Or per request:

```http
X-Router-Strategy: sticky-session-migrate
```

This combines Traefik/nginx-style sticky affinity with health-gated failover,
while the stored migrated pin provides controlled continuity instead of
per-request flapping.
