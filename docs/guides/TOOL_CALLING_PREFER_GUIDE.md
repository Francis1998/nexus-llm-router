# Tool-Calling-Prefer Routing Guide

Use `tool-calling-prefer` to bias selection toward models that support
tool / function calling when a request declares `metadata.requires_tools`
or supplies a non-empty `metadata.tools` list, for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

![Tool calling prefer demo](../../assets/tool-calling-prefer.gif)

## When to use it

- Gateways that mirror OpenRouter tool-use routing and want tool-capable
  models preferred for agentic / function-calling workloads.
- Workloads that set `requires_tools` or pass a `tools` array upstream and
  still want quality-first routing when those signals are absent.
- Fleets that maintain a per-request `metadata.tool_capable_models` allowlist
  or `metadata.model_capabilities` override for models whose tool support is
  not yet reflected in the built-in map.

## How it works

1. Read `metadata.requires_tools`. Truthy values are `true` / `1` / `yes` /
   `on`, or any other non-empty non-falsy token. Alternatively, a non-empty
   `metadata.tools` list/string also triggers preference.
2. Filter domain-eligible candidates through provider circuit health
   (emergency-retain the full eligible pool when every circuit is open).
3. When tools are required, resolve tool support from
   `metadata.tool_capable_models`, then `metadata.model_capabilities` /
   the built-in known-model map (`tools` capability). When neither is
   present for a model, treat names containing `gpt-5`, `claude`,
   `gemini`, or `kimi` as tool-capable.
4. Rank by `(supports_tools desc, quality desc, cost asc)`.
5. When the tool-calling signal is absent, route quality-first among
   healthy domain-eligible candidates.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=tool-calling-prefer
```

Or select it per request:

```http
X-Router-Strategy: tool-calling-prefer
```

```json
{
  "metadata": {
    "requires_tools": true,
    "tool_capable_models": ["kimi-k2"],
    "model_capabilities": {
      "kimi-k2": "tools"
    }
  }
}
```

No additional `NEXUS_*` environment variables are required — selection is
driven entirely by request metadata.
