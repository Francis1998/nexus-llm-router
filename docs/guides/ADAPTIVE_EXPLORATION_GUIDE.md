# Adaptive-Exploration Routing Guide

Use the `adaptive-exploration` strategy when you want epsilon-greedy explore/exploit behavior that starts exploring more aggressively while `SuccessStats` are cold, then decays toward a small residual explore rate as successes accumulate — similar to bandit schedules in popular LLM gateways.

## When to use it

- You want early traffic to sample GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2 more broadly than fixed `epsilon-greedy`.
- After providers prove reliable, you want most traffic on the highest-quality domain-eligible model with only a small residual explore budget.
- You need deterministic, replayable decisions (same `request_id` + same success window → same arm).

## How it works

1. Read `total_successes` from shared `SuccessStats` (sum of recorded provider successes).
2. Compute live epsilon:
   `epsilon = NEXUS_ADAPTIVE_EXPLORATION_MIN + (NEXUS_ADAPTIVE_EXPLORATION_BASE - NEXUS_ADAPTIVE_EXPLORATION_MIN) / (1 + total_successes)`.
3. Hash `request_id` the same way canary/A/B/`epsilon-greedy` do: `bucket = int(sha256(...)[:8], 16) / 0xFFFFFFFF`.
4. If `bucket < epsilon`, **explore**: pick uniformly among domain-eligible candidates via a second hash of `"{request_id}:explore"`.
5. Otherwise **exploit**: pick the highest `quality_score` among domain-eligible candidates.

Defaults: base `0.2` (cold start), min `0.02` (asymptotic floor).

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=adaptive-exploration
export NEXUS_ADAPTIVE_EXPLORATION_BASE=0.2
export NEXUS_ADAPTIVE_EXPLORATION_MIN=0.02
```

Or per request:

```http
X-Router-Strategy: adaptive-exploration
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
