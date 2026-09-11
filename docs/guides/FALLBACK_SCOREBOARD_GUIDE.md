# Provider Fallback Scoreboard Guide

Per-provider health scoring for fallback ordering — track success, latency,
and error rates, then recommend a fallback preference list for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

![Fallback scoreboard demo](../../assets/demo/fallback-scoreboard.gif)

## Why

LiteLLM and OpenRouter expose provider health signals used when building
failover chains. Nexus already has circuit breakers and `SuccessStats` inside
routing strategies; `ProviderFallbackScoreboard` is the reusable library
building block that ranks providers by a composite health score so the router engine (and ops tooling) can ask for a recommended
fallback order.

## How it works

1. `record_outcome(provider, success, latency_ms)` updates rolling counters.
2. Optional exponential decay (`decay_half_life_seconds`) down-weights stale
   observations using an injectable `clock` (defaults to `time.monotonic`).
3. Health score = `success_rate / (1 + avg_latency_ms / latency_ref_ms)`.
4. `rank()` / `rank(candidates)` returns providers best-first (unknown
   candidates sort last, preserving input order among ties/unknowns).
5. `snapshot(provider)` exposes success/error rates, average latency, and score.

## Example

```python
from router.fallback_scoreboard import ProviderFallbackScoreboard

board = ProviderFallbackScoreboard(decay_half_life_seconds=60.0, latency_ref_ms=1000.0)
board.record_outcome("openai", True, 120.0)
board.record_outcome("anthropic", True, 180.0)
board.record_outcome("google", False, 900.0)
board.record_outcome("moonshot", True, 250.0)

print(board.rank())
# e.g. ['openai', 'anthropic', 'moonshot', 'google']

print(board.rank(["google", "openai", "anthropic"]))
print(board.snapshot("openai"))
```


## Engine wiring

`NexusRouter` owns a process-local `ProviderFallbackScoreboard` and uses it on
every `complete` / `complete_stream` dispatch:

1. After the strategy chooses a primary model + fallback chain, the engine keeps
   the primary model first and reorders the remaining fallbacks with
   `scoreboard.rank(providers)` (healthiest provider first; unknown providers
   keep relative input order).
2. Each real provider attempt records `record_outcome(provider, success, latency_ms)`.
3. Client-side guardrail skips (budget cap, open circuit) are **not** recorded —
   those are not provider faults.

Cold start (empty scoreboard) preserves the strategy's original fallback order,
so existing routing behavior is unchanged until outcomes accumulate.

## Gap vs LiteLLM / OpenRouter

| Capability | LiteLLM / OpenRouter | Nexus |
| --- | --- | --- |
| Provider health scoring | Yes | `ProviderFallbackScoreboard` |
| Success + latency blend | Yes | Composite health score |
| Error-rate visibility | Yes | `snapshot(...).error_rate` |
| Fallback preference order | Yes | `rank()` / `rank(candidates)` |
| Injectable clock + decay | Varies | `clock=` + optional half-life |
| Frontier model traffic | Yes | GPT-5.5 / Sonnet 4.6 / Gemini 3.x / Kimi K2 |
