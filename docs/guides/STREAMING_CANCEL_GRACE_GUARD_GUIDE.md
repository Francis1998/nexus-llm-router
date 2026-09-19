# StreamingCancelGraceGuard Guide

![StreamingCancelGraceGuard](../../assets/demo/streaming-cancel-grace.gif)

Post-cancel grace window for streaming responses (`open` / `grace` / `closed`).
Allows a short final-chunk window after client abort.

Closes the Portkey / LiteLLM / OpenRouter cancel-grace gap for
**GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `StreamingBackpressureAdvisor` and `FirstTokenLatencySloAdvisor`.

## Usage

See `tests/test_cancel_grace.py`.
