# embeddings-prefer Guide

![demo](../../assets/embeddings-prefer.gif)

Prefer models that advertise `embeddings` when the request signals need it.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalogs.

## Why

OpenAI Embeddings API and LiteLLM/Portkey embedding-model routing. Gap fill vs LiteLLM/OpenRouter embedding-model preference routing.

## Signals

Truthiness on any of:
- `metadata.requires_embeddings`
- `metadata.embeddings`
- `metadata.embedding`

Optional allowlist: `metadata.embeddings_models`.

## Capability resolution

1. Allowlist (if provided)
2. `metadata.model_capabilities` / known-model map looking for `embeddings`
3. Name heuristic tokens: 'embedding', 'embeddings', 'text-embedding'

When the signal is absent, selection stays quality-first.
