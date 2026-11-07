# transcription-prefer Guide

![demo](../../assets/transcription-prefer.gif)

Prefer models that advertise `transcription` when the request signals need it.
Works with GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro catalogs.

## Why

OpenAI Whisper/transcription + Gemini audio capability routing. Gap fill vs OpenRouter/LiteLLM/Portkey capability routing.

## Signals

Truthiness on any of:
- `metadata.requires_transcription`
- `metadata.transcription`
- `metadata.speech_to_text`
- `metadata.stt`

Optional allowlist: `metadata.transcription_models`.

## Capability resolution

1. Allowlist (if provided)
2. `metadata.model_capabilities` / known-model map looking for `transcription`
3. Name heuristic tokens: 'whisper', 'transcription', 'speech-to-text', 'stt'

When the signal is absent, selection stays quality-first.
