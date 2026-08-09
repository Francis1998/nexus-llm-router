# Queue-Depth-Fairness Routing Guide

![Queue depth fairness flow](../../assets/queue-depth-fairness.gif)

Use the `queue-depth-fairness` strategy when live provider queue / in-flight
depth should steer traffic away from saturated backends so tenants share
capacity fairly.

## When to use it

- GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic needs
  quality-first routing without piling onto one already-deep provider queue.
- You already track live attempts in shared `InflightStats`.
- Operators want a soft fairness threshold
  (`NEXUS_QUEUE_DEPTH_SOFT_CAP`, default `4`) instead of only absolute
  least-busy ranking.

## How it works

1. Filter to domain-eligible catalog candidates.
2. Treat each provider's live `InflightStats.load_score` as queue depth.
3. Among providers under the soft cap, select the highest-quality model.
4. If every provider is at or above the soft cap, select the lowest depth,
   then highest quality.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=queue-depth-fairness
export NEXUS_QUEUE_DEPTH_SOFT_CAP=4
```

Or per request:

```http
X-Router-Strategy: queue-depth-fairness
```

## Tuning notes

- Lower the soft cap (for example `2`) when queues should shed early under
  multi-tenant load.
- Raise the soft cap (for example `8`) when providers can absorb deeper
  bursts before fairness shedding matters.
- Pair with `tenant-concurrency-lease` for per-tenant lease fairness, or
  `least-busy` when absolute lowest depth should always win.

## Suggested repo metadata

- **Description:** Multi-provider LLM router with pluggable strategies, safety
  guardrails, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog.
- **Topics:** `llm-router`, `litellm`, `openai`, `anthropic`, `gemini`, `routing`, `python`
