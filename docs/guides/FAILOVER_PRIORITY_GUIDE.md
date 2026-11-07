# Failover-Priority Routing Guide

Use the `failover-priority` strategy when you want LiteLLM-style ordered failover:
an explicit preference list of models, walking down until a healthy provider is
found.

## When to use it

- Operators define a hard preference order (for example GPT-6 Astra → Claude Opus
  5.5 → Gemini 4 Argon → Kimi K3.1 Pro) rather than optimizing for cost or quality.
- Open circuit breakers should skip a preference without changing the ordered
  intent of the remaining list.
- You need a deterministic fallback chain that preserves that same order.

## How it works

1. Resolve `NEXUS_FAILOVER_PRIORITY` to catalog models (unknown names are ignored
   as long as at least one remains).
2. Walk the list in order and pick the first model whose provider circuit is
   closed (`ProviderHealth.is_available`).
3. If every preference is unhealthy, route to the first listed catalog model so
   decide-time never hard-fails.
4. Build the fallback chain from the remaining preference order.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=failover-priority
export NEXUS_FAILOVER_PRIORITY='["gpt-6-astra","claude-opus-5-5","gemini-4-argon","kimi-k3.1-pro"]'
```

Or per request:

```http
X-Router-Strategy: failover-priority
```

## Demo

![Failover-priority routing demo](../../assets/failover-priority.gif)
