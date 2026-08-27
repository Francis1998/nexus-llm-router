# Structured-Output-Prefer Routing Guide

Use `structured-output-prefer` to bias selection toward models that advertise
JSON / structured-output support when a request declares
`metadata.requires_json` or `metadata.structured_output`, for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

![Structured output prefer demo](../../assets/structured-output-prefer.gif)

## When to use it

- Gateways that mirror LiteLLM / OpenRouter / Portkey structured-output
  routing and want JSON-capable models preferred for schema-constrained
  responses.
- Workloads that set `requires_json` / `structured_output` upstream and still
  want quality-first routing when those flags are absent.
- Fleets that maintain a per-request `metadata.structured_models` allowlist
  or `metadata.model_capabilities` override for models whose JSON support is
  not yet reflected in the built-in map.

## How it works

1. Read `metadata.requires_json` or `metadata.structured_output`. Truthy
   values are `true` / `1` / `yes` / `on`, or any other non-empty non-falsy
   token.
2. Filter domain-eligible candidates through provider circuit health
   (emergency-retain the full eligible pool when every circuit is open).
3. When structured output is requested, resolve capability from
   `metadata.structured_models`, then `metadata.model_capabilities` /
   the built-in known-model map (`json`, `structured`, or `json_mode`).
   When neither is present for a model, treat names containing `gpt-5`,
   `claude`, `gemini`, or `kimi` as structured-capable.
4. Rank by `(supports_structured desc, quality desc, cost asc)`.
5. When the structured-output signal is absent, route quality-first among
   healthy domain-eligible candidates.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=structured-output-prefer
```

Or select it per request:

```http
X-Router-Strategy: structured-output-prefer
```

```json
{
  "metadata": {
    "requires_json": true,
    "structured_models": ["kimi-k2"],
    "model_capabilities": {
      "kimi-k2": "json"
    }
  }
}
```

No additional `NEXUS_*` environment variables are required — selection is
driven entirely by request metadata.
