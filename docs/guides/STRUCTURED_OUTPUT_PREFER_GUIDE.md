# Structured-Output-Prefer Routing Guide

Use `structured-output-prefer` to bias selection toward models that advertise
JSON / structured-output support when a request declares
`metadata.structured_output` or `metadata.json_mode`, for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

![Structured output prefer demo](../../assets/structured-output-prefer.gif)

## When to use it

- Gateways that mirror LiteLLM / OpenRouter structured-output routing and
  want JSON-capable models preferred for schema-constrained responses.
- Workloads that set `json_mode` or `structured_output` upstream and still
  want quality-first routing when those flags are absent.
- Fleets that maintain a per-request `metadata.model_capabilities` override
  for models whose JSON support is not yet reflected in the built-in map.

## How it works

1. Read `metadata.structured_output` or `metadata.json_mode`. Truthy values
   are `true` / `1` / `yes` / `on`, or any other non-empty non-falsy token.
2. Filter domain-eligible candidates through provider circuit health
   (emergency-retain the full eligible pool when every circuit is open).
3. When structured output is requested, resolve each candidate's capability
   set from `metadata.model_capabilities` or the built-in known-model map.
   When neither is present for a model, treat names containing `gpt-5`,
   `claude`, `gemini`, or `kimi` as JSON-capable.
4. Rank by `(has_json desc, quality desc, cost asc)`.
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
    "json_mode": true,
    "model_capabilities": {
      "kimi-k2": "json"
    }
  }
}
```

No additional `NEXUS_*` environment variables are required — selection is
driven entirely by request metadata.
