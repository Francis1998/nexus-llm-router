# Prompt-Length-Tier-Shed Routing Guide

![Prompt-length tier shed flow](../../assets/prompt-length-tier-shed.gif)

Use the `prompt-length-tier-shed` strategy when LiteLLM/OpenRouter-style traffic
needs a **prompt-length tier gate**: long prompts shed frontier models so spend
stays on mid/economy SKUs unless no alternative exists.

## When to use it

- Long context / RAG prompts should not default to frontier GPT-5.5 /
  Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 when a mid-tier model is enough.
- You want a soft length-based shed, complementary to `token-budget` and
  `complexity-tier`.
- Short hard prompts should still reach frontier quality.

## How it works

1. Filter domain-eligible catalog candidates.
2. Read `signals.prompt_tokens_estimate`.
3. When prompt tokens are **within** `NEXUS_PROMPT_LENGTH_TIER_TOKENS`, pick the
   highest-quality eligible model (frontier allowed).
4. When prompt tokens are **above** the gate, shed frontier-tier models and pick
   the highest-quality mid/economy candidate (cost breaks ties). If every
   eligible model is frontier, fall back to quality ranking.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=prompt-length-tier-shed
export NEXUS_PROMPT_LENGTH_TIER_TOKENS=8000
```

Or per request:

```http
X-Router-Strategy: prompt-length-tier-shed
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).

## Suggested repo metadata

- **Description:** Multi-provider LLM router with pluggable strategies, safety
  guardrails, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog.
- **Topics:** `llm-router`, `litellm`, `openai`, `anthropic`, `gemini`, `routing`, `python`
