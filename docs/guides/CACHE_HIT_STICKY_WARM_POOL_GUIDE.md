# Cache-Hit-Sticky-Warm-Pool Routing Guide

![Cache-hit sticky warm pool flow](../../assets/cache-hit-sticky-warm-pool.gif)

Use the `cache-hit-sticky-warm-pool` strategy when LiteLLM/OpenRouter-style
traffic should **reuse a warm provider prompt cache** by pinning identical long
prefixes onto one model.

## When to use it

- Repeated system prompts / tool schemas should stay on one model for cache hits.
- You want sticky affinity across GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
  Kimi K2 without requiring an explicit session id.
- Unhealthy sticky primaries should failover deterministically.

## How it works

1. Filter domain-eligible catalog candidates.
2. Build a sticky key from leading message content once it reaches
   `NEXUS_CACHE_HIT_STICKY_MIN_CHARS` (else fall back to `session_id`).
3. Consistent-hash the key onto the ordered eligible catalog.
4. Walk a healthy failover ring when the primary provider is unavailable.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=cache-hit-sticky-warm-pool
export NEXUS_CACHE_HIT_STICKY_MIN_CHARS=64
```

Or per request:

```http
X-Router-Strategy: cache-hit-sticky-warm-pool
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).

## Suggested repo metadata

- **Description:** Multi-provider LLM router with pluggable strategies, safety
  guardrails, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog.
- **Topics:** `llm-router`, `litellm`, `openai`, `anthropic`, `gemini`, `routing`, `python`
