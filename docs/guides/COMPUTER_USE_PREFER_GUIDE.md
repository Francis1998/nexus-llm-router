# Computer-Use-Prefer Routing Guide

Use `computer-use-prefer` to bias selection toward `computer_use`-capable
models when a request declares `metadata.requires_computer_use` / `metadata.computer_use` / `metadata.cua`, for GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.

![computer-use-prefer demo](../../assets/computer-use-prefer.gif)

## When to use it

- Gateways that mirror OpenRouter / LiteLLM / Portkey computer-use capability routing and want
  `computer_use`-capable models preferred when the signal is set.
- Workloads that set `requires_computer_use`, `computer_use`, `cua` upstream and still want quality-first
  routing when those flags are absent.
- Fleets that maintain a per-request `metadata.computer_use_models` allowlist
  or `metadata.model_capabilities` override for models whose
  `computer_use` support is not yet reflected in the built-in map.

## How it works

1. Read `metadata.requires_computer_use` / `metadata.computer_use` / `metadata.cua`. Truthy values are `true` / `1` / `yes` / `on`, or any
   other non-empty non-falsy token.
2. Filter domain-eligible candidates through provider circuit health
   (emergency-retain the full eligible pool when every circuit is open).
3. When the signal is present, resolve capability from
   `metadata.computer_use_models`, then `metadata.model_capabilities` /
   the built-in known-model map (`computer_use`). When neither is
   present for a model, treat names containing `computer`, `cua`, `operator` as capable.
4. Rank by `(supports_computer_use desc, quality desc, cost asc)`.
5. When the signal is absent, route quality-first among healthy
   domain-eligible candidates.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=computer-use-prefer
```

Or select it per request:

```http
X-Router-Strategy: computer-use-prefer
```

```json
{
  "metadata": {
    "requires_computer_use": true,
    "computer_use_models": ["gemini-3.5-flash"],
    "model_capabilities": {
      "gemini-3.5-flash": "computer_use"
    }
  }
}
```

No additional `NEXUS_*` environment variables are required — selection is
driven entirely by request metadata.
