# Streaming-Prefer Routing Guide

Use `streaming-prefer` to bias selection toward streaming-capable models when
a request declares `metadata.stream` or `metadata.requires_streaming`, for
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

![Streaming prefer demo](../../assets/streaming-prefer.gif)

## When to use it

- Gateways that mirror OpenRouter / LiteLLM / Portkey streaming capability
  routing and want stream-capable models preferred for SSE / token streaming.
- Workloads that set `stream` / `requires_streaming` upstream and still want
  quality-first routing when those flags are absent.
- Fleets that maintain a per-request `metadata.streaming_models` allowlist
  or `metadata.model_capabilities` override for models whose streaming
  support is not yet reflected in the built-in map.

## How it works

1. Read `metadata.stream` or `metadata.requires_streaming`. Truthy values are
   `true` / `1` / `yes` / `on`, or any other non-empty non-falsy token.
2. Filter domain-eligible candidates through provider circuit health
   (emergency-retain the full eligible pool when every circuit is open).
3. When streaming is requested, resolve capability from
   `metadata.streaming_models`, then `metadata.model_capabilities` /
   the built-in known-model map (`streaming`). When neither is present for a
   model, treat names containing `gpt-5`, `claude`, `gemini`, or `kimi` as
   streaming-capable.
4. Rank by `(supports_streaming desc, quality desc, cost asc)`.
5. When the streaming signal is absent, route quality-first among healthy
   domain-eligible candidates.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=streaming-prefer
```

Or select it per request:

```http
X-Router-Strategy: streaming-prefer
```

```json
{
  "metadata": {
    "stream": true,
    "streaming_models": ["gemini-3.5-flash"],
    "model_capabilities": {
      "gemini-3.5-flash": "streaming"
    }
  }
}
```

No additional `NEXUS_*` environment variables are required — selection is
driven entirely by request metadata.
