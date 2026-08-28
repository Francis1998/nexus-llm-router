# Web-Search-Prefer Routing Guide

Use `web-search-prefer` to bias selection toward `web_search`-capable
models when a request declares `metadata.requires_web_search` / `metadata.web_search` / `metadata.online`, for GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.

![web-search-prefer demo](../../assets/web-search-prefer.gif)

## When to use it

- Gateways that mirror OpenRouter / LiteLLM / Portkey web-search capability routing and want
  `web_search`-capable models preferred when the signal is set.
- Workloads that set `requires_web_search`, `web_search`, `online` upstream and still want quality-first
  routing when those flags are absent.
- Fleets that maintain a per-request `metadata.web_search_models` allowlist
  or `metadata.model_capabilities` override for models whose
  `web_search` support is not yet reflected in the built-in map.

## How it works

1. Read `metadata.requires_web_search` / `metadata.web_search` / `metadata.online`. Truthy values are `true` / `1` / `yes` / `on`, or any
   other non-empty non-falsy token.
2. Filter domain-eligible candidates through provider circuit health
   (emergency-retain the full eligible pool when every circuit is open).
3. When the signal is present, resolve capability from
   `metadata.web_search_models`, then `metadata.model_capabilities` /
   the built-in known-model map (`web_search`). When neither is
   present for a model, treat names containing `search`, `online`, `browse` as capable.
4. Rank by `(supports_web_search desc, quality desc, cost asc)`.
5. When the signal is absent, route quality-first among healthy
   domain-eligible candidates.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=web-search-prefer
```

Or select it per request:

```http
X-Router-Strategy: web-search-prefer
```

```json
{
  "metadata": {
    "requires_web_search": true,
    "web_search_models": ["gemini-3.5-flash"],
    "model_capabilities": {
      "gemini-3.5-flash": "web_search"
    }
  }
}
```

No additional `NEXUS_*` environment variables are required — selection is
driven entirely by request metadata.
