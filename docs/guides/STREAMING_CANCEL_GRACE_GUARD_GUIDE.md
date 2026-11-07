# StreamingCancelGraceGuard Guide

![StreamingCancelGraceGuard](../../assets/demo/streaming-cancel-grace.gif)

Post-cancel grace window for streaming responses (`open` / `grace` / `closed`).
Allows a short final-chunk window after client abort.

Closes the Portkey / LiteLLM / OpenRouter cancel-grace gap for
**GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `StreamingBackpressureAdvisor` and `FirstTokenLatencySloAdvisor`.

## Usage

See `tests/test_cancel_grace.py`.
