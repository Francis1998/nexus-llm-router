# Long-Context-Prefer Routing Guide

Use `long-context-prefer` to bias selection toward models whose context
window meets a requested threshold when a request declares
`metadata.min_context_tokens` or `metadata.long_context`, for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

![Long context prefer demo](../../assets/long-context-prefer.gif)

## When to use it

- Gateways that mirror OpenRouter / LiteLLM / Portkey long-context routing
  and want large-window models preferred for long prompts or RAG packs.
- Workloads that set `min_context_tokens` / `long_context` upstream and still
  want quality-first routing when those signals are absent.
- Fleets that maintain a per-request `metadata.model_context_windows` map
  for models whose windows are not yet reflected in the built-in map.

## How it works

1. Read `metadata.min_context_tokens` (positive int) or
   `metadata.long_context` (truthy → default threshold `100000`).
2. Filter domain-eligible candidates through provider circuit health
   (emergency-retain the full eligible pool when every circuit is open).
3. Resolve each candidate's context window from
   `metadata.model_context_windows`, the built-in known-model map, the
   catalog `context_window`, or a name heuristic matching `gpt-5`,
   `claude`, `gemini`, or `kimi` (default `200000`).
4. Rank by `(meets_threshold desc, context_window desc, quality desc,
   cost asc)`.
5. When the long-context signal is absent, route quality-first among
   healthy domain-eligible candidates.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=long-context-prefer
```

Or select it per request:

```http
X-Router-Strategy: long-context-prefer
```

```json
{
  "metadata": {
    "min_context_tokens": 500000,
    "model_context_windows": {
      "gemini-3.1-pro-preview": 1000000
    }
  }
}
```

No additional `NEXUS_*` environment variables are required — selection is
driven entirely by request metadata.
