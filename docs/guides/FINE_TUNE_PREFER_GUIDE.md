# fine-tune-prefer Guide

![demo](../../assets/fine-tune-prefer.gif)

Prefer models that advertise `fine_tune` when the request signals need it.
Works with GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro catalogs.

## Why

OpenAI Fine-tuning API and LiteLLM/Portkey fine-tune model routing. Gap fill vs LiteLLM/OpenRouter fine-tuned deployment routing.

## Signals

Truthiness on any of:
- `metadata.requires_fine_tune`
- `metadata.fine_tune`
- `metadata.ft_model`

Optional allowlist: `metadata.fine_tune_models`.

> **HTTP note:** strategies read `RouterRequest.metadata`. `POST /v1/chat/completions`,
> `/v1/responses`, and `/v1/messages` do not forward request-body `metadata` to routing
> yet, so over HTTP this strategy takes its no-signal path. Set `metadata` on
> `RouterRequest` when embedding `NexusRouter`, as the demo does
> (`scripts/render_strategy_demos.py`).

## Capability resolution

1. Allowlist (if provided)
2. `metadata.model_capabilities` / known-model map looking for `fine_tune`
3. Name heuristic tokens: 'fine-tune', 'finetune', 'ft:'

When the signal is absent, selection stays quality-first.
