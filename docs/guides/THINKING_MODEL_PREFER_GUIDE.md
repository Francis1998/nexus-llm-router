# Thinking-Model-Prefer Routing Guide

Use `thinking-model-prefer` to bias selection toward reasoning / thinking
models when task complexity is at or above
`NEXUS_THINKING_COMPLEXITY_THRESHOLD`, for GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.

![Thinking model prefer demo](../../assets/thinking-model-prefer.gif)

## When to use it

- Gateways that mirror LiteLLM reasoning-model routing and want
  thinking-capable models preferred for hard tasks.
- Workloads that already score complexity upstream via
  `metadata.complexity_score` (or rely on the observe-phase
  `signals.complexity_score`) and still want quality-first routing below
  the threshold.
- Fleets that maintain a per-request `metadata.thinking_models` allowlist
  for models whose reasoning SKUs are not covered by the name heuristic.

## How it works

1. Resolve complexity from `metadata.complexity_score` when present and
   numeric (clamped to `[0.0, 1.0]`); otherwise use
   `signals.complexity_score`. Malformed values fall back to signals.
2. Filter domain-eligible candidates through provider circuit health
   (emergency-retain the full eligible pool when every circuit is open).
3. When complexity is at or above `NEXUS_THINKING_COMPLEXITY_THRESHOLD`
   (default `0.7`), resolve thinking membership from
   `metadata.thinking_models` when provided; otherwise treat model names
   containing `o1`, `o3`, `reasoning`, `thinking`, `sonnet`, or `opus` as
   thinking-capable.
4. Rank by `(is_thinking desc, quality desc, cost asc)`.
5. Below the threshold, route quality-first among healthy domain-eligible
   candidates.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=thinking-model-prefer
export NEXUS_THINKING_COMPLEXITY_THRESHOLD=0.7
```

Or select it per request:

```http
X-Router-Strategy: thinking-model-prefer
```

```json
{
  "metadata": {
    "complexity_score": 0.85,
    "thinking_models": ["claude-sonnet-4-6", "o3"]
  }
}
```
