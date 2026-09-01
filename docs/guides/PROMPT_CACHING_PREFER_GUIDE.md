# prompt-caching-prefer Guide

![demo](../../assets/prompt-caching-prefer.gif)

Prefer models that advertise `prompt_caching` when the request signals need it.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalogs.

## Why

Anthropic/OpenAI prompt caching and Portkey cache-aware routing. Gap fill vs Portkey/LiteLLM prompt-cache capability routing.

## Signals

Truthiness on any of:
- `metadata.requires_prompt_caching`
- `metadata.prompt_caching`
- `metadata.cache_prompt`

Optional allowlist: `metadata.prompt_caching_models`.

## Capability resolution

1. Allowlist (if provided)
2. `metadata.model_capabilities` / known-model map looking for `prompt_caching`
3. Name heuristic tokens: 'prompt-caching', 'prompt-cache', 'cached-tokens'

When the signal is absent, selection stays quality-first.
