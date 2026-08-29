# code-execution-prefer Guide

![demo](../../assets/code-execution-prefer.gif)

Prefer models that advertise `code_execution` when the request signals need it.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalogs.

## Why

OpenRouter / LiteLLM / Portkey code-execution capability routing. Gap fill vs OpenRouter/LiteLLM/Portkey capability routing.

## Signals

Truthiness on any of:
- `metadata.requires_code_execution`
- `metadata.code_execution`
- `metadata.code_interpreter`

Optional allowlist: `metadata.code_execution_models`.

## Capability resolution

1. Allowlist (if provided)
2. `metadata.model_capabilities` / known-model map looking for `code_execution`
3. Name heuristic tokens: 'code-execution', 'code_interpreter', 'interpreter', 'sandbox'

When the signal is absent, selection stays quality-first.
