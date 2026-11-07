# Token-Budget Routing Guide

Use the `token-budget` strategy when prompts (or completion caps) can approach a model's context window and you want the highest-quality domain-eligible model that still fits.

## When to use it

- Long documents, RAG contexts, or large `max_tokens` risk exceeding smaller windows.
- You want quality first among GPT-6, Claude 5.5, Gemini 4 / 3.8, and Kimi K3.1 — but only models that can hold the estimate.
- You need a hard fail-soft path: if nothing fits, Nexus still routes to the eligible model that accepts the largest prompt.

## How it works

1. Estimate tokens needed as `prompt_tokens_estimate + max_tokens`.
2. For each domain-eligible candidate, compute effective capacity as `min(context_window, request.token_budget)`.
3. Keep candidates where `tokens_needed <= effective capacity` and the prompt alone fits the model's maximum input (`ModelCandidate.input_token_limit`).
4. Among those, pick the highest `quality_score` (ties break toward larger context).
5. If none fit, fall back to the eligible model that accepts the largest prompt (ties break toward larger context, then quality).

Catalog priors include context windows: GPT-6 Astra / Luna 1,050,000 with at most 922,000 input tokens (`max_input_tokens`), Claude Opus / Haiku 5.5 1,000,000, and Gemini 4 Argon / 3.8 Flash / Kimi K3.1 Pro 1,048,576. A 1,020,000-token prompt therefore routes to Gemini 4 Argon, not GPT-6 Astra.

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
