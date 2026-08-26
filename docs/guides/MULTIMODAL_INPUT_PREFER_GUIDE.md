# Multimodal-Input-Prefer Routing Guide

Use `multimodal-input-prefer` to bias selection toward vision /
multimodal-capable models when a request declares `metadata.has_images`
or `metadata.has_audio`, for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.

![Multimodal input prefer demo](../../assets/multimodal-input-prefer.gif)

## When to use it

- Gateways that mirror OpenRouter multimodal routing and want
  vision-capable models preferred for image or audio inputs.
- Workloads that set `has_images` / `has_audio` upstream and still want
  quality-first routing when those flags are absent.
- Fleets that maintain a per-request `metadata.multimodal_models` allowlist
  or `metadata.model_capabilities` override for models whose vision support
  is not yet reflected in the built-in map.

## How it works

1. Read `metadata.has_images` or `metadata.has_audio`. Truthy values are
   `true` / `1` / `yes` / `on`, or any other non-empty non-falsy token.
2. Filter domain-eligible candidates through provider circuit health
   (emergency-retain the full eligible pool when every circuit is open).
3. When multimodal input is present, resolve capability from
   `metadata.multimodal_models`, then `metadata.model_capabilities` /
   the built-in known-model map (`vision`, `audio`, or `multimodal`).
   When neither is present for a model, treat names containing `gpt-5`,
   `claude`, `gemini`, or `vision` as multimodal-capable.
4. Rank by `(supports_multimodal desc, quality desc, cost asc)`.
5. When the multimodal signal is absent, route quality-first among
   healthy domain-eligible candidates.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=multimodal-input-prefer
```

Or select it per request:

```http
X-Router-Strategy: multimodal-input-prefer
```

```json
{
  "metadata": {
    "has_images": true,
    "multimodal_models": ["gemini-3.5-flash"],
    "model_capabilities": {
      "gemini-3.5-flash": "vision"
    }
  }
}
```

No additional `NEXUS_*` environment variables are required — selection is
driven entirely by request metadata.
