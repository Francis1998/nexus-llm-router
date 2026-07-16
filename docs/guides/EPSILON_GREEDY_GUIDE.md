# Epsilon-Greedy Routing Guide

Use the `epsilon-greedy` strategy when you want most traffic on the highest-quality domain-eligible model, while still sampling the rest of the catalog so new or under-prioritized models get live feedback.

## When to use it

- You trust quality priors for day-to-day routing but want a small explore budget.
- You are evaluating catalog changes across GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2 without running a full A/B experiment.
- You need deterministic, replayable decisions (same `request_id` → same arm).

## How it works

1. Hash `request_id` the same way canary/A/B do: `bucket = int(sha256(...)[:8], 16) / 0xFFFFFFFF`.
2. If `bucket < NEXUS_EPSILON`, **explore**: pick uniformly among domain-eligible candidates via a second hash of `"{request_id}:explore"`.
3. Otherwise **exploit**: pick the highest `quality_score` among domain-eligible candidates.

Default `NEXUS_EPSILON` is `0.1` (10% explore).

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=epsilon-greedy
export NEXUS_EPSILON=0.1
```

Or per request:

```http
X-Router-Strategy: epsilon-greedy
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
