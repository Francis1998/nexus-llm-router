# pdf-input-prefer Guide

![demo](../../assets/pdf-input-prefer.gif)

Prefer models that advertise `pdf` when the request signals need it.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalogs.

## Why

OpenRouter / LiteLLM / Portkey video-input capability routing. Gap fill vs OpenRouter/LiteLLM/Portkey capability routing.

## Signals

Truthiness on any of:
- `metadata.requires_pdf`
- `metadata.pdf_input`
- `metadata.has_pdf`

Optional allowlist: `metadata.pdf_models`.

## Capability resolution

1. Allowlist (if provided)
2. `metadata.model_capabilities` / known-model map looking for `pdf`
3. Name heuristic tokens: 'video', 'veo', 'gemini-video'

When the signal is absent, selection stays quality-first.
