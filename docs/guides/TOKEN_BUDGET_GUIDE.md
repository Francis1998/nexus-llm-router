# Token-Budget Routing Guide

Use the `token-budget` strategy when prompts (or completion caps) can approach a model's context window and you want the highest-quality domain-eligible model that still fits.

## When to use it

- Long documents, RAG contexts, or large `max_tokens` risk exceeding smaller windows.
- You want quality first among GPT-5.5, Claude Sonnet 4.6, Gemini 2.5, and Kimi K2 — but only models that can hold the estimate.
- You need a hard fail-soft path: if nothing fits, Nexus still routes to the largest-context eligible model.

## How it works

1. Estimate tokens needed as `prompt_tokens_estimate + max_tokens`.
2. For each domain-eligible candidate, compute effective capacity as `min(context_window, request.token_budget)`.
3. Keep candidates where `tokens_needed <= effective capacity`.
4. Among those, pick the highest `quality_score` (ties break toward larger context).
5. If none fit, fall back to the largest-context eligible model.

Catalog priors include context windows (for example Claude Sonnet 4.6 / GPT-5.5 at 200k, Gemini 2.5-class at 1M, Kimi K2 at 128k).

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=token-budget
```

Or per request:

```http
X-Router-Strategy: token-budget
```

Set `token_budget` on the router request (default `4096`) to cap how much of each model's window you are willing to use.

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
