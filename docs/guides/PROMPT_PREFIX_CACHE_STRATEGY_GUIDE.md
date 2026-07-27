# Prompt-Prefix-Cache Routing Guide

Use the `prompt-prefix-cache` strategy when many requests share a long reusable
system prompt and you want those requests to keep landing on the same
provider/model. This improves the odds of provider-side KV-cache hits in
OpenRouter-, LiteLLM-, and native prompt-caching deployments.

## When to use it

- Agents or RAG services prepend a long, stable system prompt to each request.
- You want GPT-5.5, Claude Sonnet 4.6, Gemini 2.5, and Kimi K2 traffic to keep
  shared prefixes sticky to one provider/model instead of spreading each prefix
  across the fleet.
- The upstream gateway/provider supports prompt-prefix or KV-cache reuse and
  cache locality is scoped to a model deployment.

## How it works

1. Join all `system` messages in the request.
2. If the system prompt is shorter than `NEXUS_PROMPT_PREFIX_CACHE_MIN_CHARS`,
   fall back to `cost-optimal` under `NEXUS_QUALITY_FLOOR`.
3. Hash the first `NEXUS_PROMPT_PREFIX_CACHE_MIN_CHARS` characters of the system
   prompt with SHA-256.
4. Bucket that hash across domain-eligible, realtime-capable candidates ordered
   by `(provider, model)`.
5. Route every request sharing that long prefix to the same provider/model while
   unrelated prefixes spread across the eligible pool.

Only the configured prefix length is hashed. Two prompts with the same long
system-prompt prefix but different user turns or different suffix instructions
therefore keep the same cache affinity.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=prompt-prefix-cache
export NEXUS_PROMPT_PREFIX_CACHE_MIN_CHARS=512
export NEXUS_QUALITY_FLOOR=0.72
```

Or per request:

```http
X-Router-Strategy: prompt-prefix-cache
```

Example internal request:

```python
RouterRequest(
    request_id="req-1",
    messages=[
        ChatMessage(
            role="system",
            content="You are the company's support policy agent..." * 80,
        ),
        ChatMessage(role="user", content="Summarize the refund policy."),
    ],
    strategy=RoutingStrategyName.PROMPT_PREFIX_CACHE,
)
```

## Demo

![Prompt-prefix-cache routing demo](../../assets/prompt-prefix-cache.gif)
