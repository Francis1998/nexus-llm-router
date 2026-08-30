# batch-api-prefer Guide

![demo](../../assets/batch-api-prefer.gif)

Prefer models that advertise `batch` when the request signals need it.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalogs.

## Why

OpenRouter / LiteLLM / Portkey video-input capability routing. Gap fill vs OpenRouter/LiteLLM/Portkey capability routing.

## Signals

Truthiness on any of:
- `metadata.requires_batch`
- `metadata.batch_api`
- `metadata.use_batch`

Optional allowlist: `metadata.batch_models`.

## Capability resolution

1. Allowlist (if provided)
2. `metadata.model_capabilities` / known-model map looking for `batch`
3. Name heuristic tokens: 'video', 'veo', 'gemini-video'

When the signal is absent, selection stays quality-first.
