# fine-tune-prefer Guide

![demo](../../assets/fine-tune-prefer.gif)

Prefer models that advertise `fine_tune` when the request signals need it.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalogs.

## Why

OpenAI Fine-tuning API and LiteLLM/Portkey fine-tune model routing. Gap fill vs LiteLLM/OpenRouter fine-tuned deployment routing.

## Signals

Truthiness on any of:
- `metadata.requires_fine_tune`
- `metadata.fine_tune`
- `metadata.ft_model`

Optional allowlist: `metadata.fine_tune_models`.

## Capability resolution

1. Allowlist (if provided)
2. `metadata.model_capabilities` / known-model map looking for `fine_tune`
3. Name heuristic tokens: 'fine-tune', 'finetune', 'ft:'

When the signal is absent, selection stays quality-first.
