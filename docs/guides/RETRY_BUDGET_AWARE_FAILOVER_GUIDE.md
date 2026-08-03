# Retry-Budget-Aware-Failover Routing Guide

![Retry budget failover flow](../../assets/retry-budget-aware-failover.gif)

Use the `retry-budget-aware-failover` strategy when LiteLLM/OpenRouter-style
gateways track remaining retries and need **quality while budget remains**, then
**low-latency failover** on the last attempt.

## When to use it

- Upstream clients pass `metadata.retry_remaining` on each attempt.
- Early attempts should still prefer GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
  Kimi K2 quality.
- The final retry should favor the fastest healthy provider.

## How it works

1. Filter domain-eligible candidates; prefer healthy providers.
2. Resolve remaining retries from `metadata.retry_remaining`, else
   `NEXUS_RETRY_BUDGET_DEFAULT`.
3. When remaining retries are `> 1`, pick highest-quality healthy model.
4. When remaining retries are `<= 1`, failover to the lowest rolling-p95 healthy
   model.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=retry-budget-aware-failover
export NEXUS_RETRY_BUDGET_DEFAULT=3
```

Or per request:

```http
X-Router-Strategy: retry-budget-aware-failover
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).

## Suggested repo metadata

- **Description:** Multi-provider LLM router with pluggable strategies, safety
  guardrails, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog.
- **Topics:** `llm-router`, `litellm`, `openai`, `anthropic`, `gemini`, `routing`, `python`
