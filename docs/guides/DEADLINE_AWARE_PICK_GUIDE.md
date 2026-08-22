# Deadline-Aware-Pick Routing Guide

Use `deadline-aware-pick` when callers publish remaining request budget and need
the router to switch from quality-first to fastest-healthy selection under
tight deadlines for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## When to use it

- Interactive APIs with hard response SLAs.
- Cascading gateways that forward remaining timeout budgets.
- Workloads that prefer quality until the deadline becomes tight.

## How it works

1. Read `metadata.remaining_ms` or `metadata.deadline_ms`.
2. Compare the remaining budget to `NEXUS_DEADLINE_AWARE_THRESHOLD_MS`
   (default `500`).
3. Filter domain-eligible candidates through provider circuit health.
4. If remaining budget is below the threshold, pick the lowest observed healthy
   provider latency.
5. Otherwise keep quality-first routing.
6. Missing or invalid deadline metadata stays quality-first.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=deadline-aware-pick
export NEXUS_DEADLINE_AWARE_THRESHOLD_MS=500.0
```

Or select it per request:

```http
X-Router-Strategy: deadline-aware-pick
```

```json
{"metadata": {"remaining_ms": 120}}
```

```json
{"metadata": {"deadline_ms": 800}}
```
