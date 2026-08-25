# Provider-Exclusion Routing Guide

Use `provider-exclusion` to filter providers and models listed in request
metadata out of the healthy domain-eligible pool before quality-first
selection for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

![Provider exclusion demo](../../assets/provider-exclusion.gif)

## When to use it

- Gateways that mirror Portkey / Helicone provider allow/deny lists and need
  a per-request deny list without changing the default catalog.
- Multi-tenant fleets that must keep certain providers or model SKUs off a
  given customer path for compliance or cost reasons.
- Operational drain windows where a provider is still healthy but should not
  receive new primary traffic.

## How it works

1. Parse `metadata.excluded_providers` and `metadata.excluded_models` — each
   a comma-separated string or a list of names (case-insensitive).
2. Filter domain-eligible candidates through provider circuit health.
3. Drop candidates whose provider or model appears in the exclusion lists.
4. Among remaining candidates, pick the highest quality (cost as tie-break).
5. If every candidate is excluded, emergency-retain the highest-quality
   model from the full domain-eligible pool and note emergency in the
   rationale so the request still routes.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-exclusion
```

Or select it per request:

```http
X-Router-Strategy: provider-exclusion
```

```json
{
  "metadata": {
    "excluded_providers": "openai, moonshot",
    "excluded_models": ["claude-haiku-4-5"]
  }
}
```

No additional `NEXUS_*` environment variables are required — selection is
driven entirely by request metadata.
