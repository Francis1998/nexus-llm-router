# Quality-Weighted-Sticky Routing Guide

Use `quality-weighted-sticky` when GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 sessions need sticky affinity like `sticky-session`, but higher-quality
models should own a larger share of the hash ring.

![Demo](../../assets/quality-weighted-sticky.gif)

## When to use it

- Multi-turn sessions must stay on one model for prompt-cache affinity.
- Uniform sticky hashing under-utilizes higher-quality catalog entries.
- Tenant hashing (`sticky-tenant-hash`) is the wrong sticky key for conversational turns.

## How it works

1. Filter the catalog to domain-eligible models.
2. Sort candidates by model name for a stable ring order.
3. Assign each candidate an integer weight from `quality_score` (scaled ×100, min 1).
4. Hash `session_id` into the total weight range and walk cumulative weights.
5. Every request with the same `session_id` maps to the same model.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=quality-weighted-sticky
```

Or per request:

```http
X-Router-Strategy: quality-weighted-sticky
```

## Tuning notes

- Adjust catalog `quality_score` priors to reshape sticky share.
- No dedicated `NEXUS_*` knob — weights are derived from the model catalog.
- Distinct sessions still spread across the pool, biased toward higher quality.
