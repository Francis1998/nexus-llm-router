# Provider-Token-Fair-Share Routing Guide

Use `provider-token-fair-share` when prompt-token volume should be balanced
across providers in a rolling window so GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic avoids starving one backend's token quota.

![Provider token fair share demo](../../assets/provider-token-fair-share.gif)

## When to use it

- Provider token RPM limits are the bottleneck, not request counts.
- You need round-robin selection weighted by remaining token headroom.
- `provider-quota-fair-share` balances requests, but token volume is uneven.

## How it works

1. Track estimated prompt tokens per provider in a rolling 60-second window.
2. Compute remaining quota:
   `NEXUS_PROVIDER_TOKEN_FAIR_SHARE_CEILING - used - current_prompt_tokens`.
3. Prefer providers with the highest remaining quota.
4. Break ties with request-id weighted round-robin among tied providers.
5. When every provider is over quota, fall back to the least-used provider.

Unlike `token-rpm-ceiling`, this strategy actively rotates toward providers
with the most remaining fair-share headroom rather than quality-first shedding.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-token-fair-share
export NEXUS_PROVIDER_TOKEN_FAIR_SHARE_CEILING=100000
```

Or per request:

```http
X-Router-Strategy: provider-token-fair-share
```
