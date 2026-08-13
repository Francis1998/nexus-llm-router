# Latency-Slope-Shed Routing Guide

Use `latency-slope-shed` when GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 providers show a rising EWMA latency trend and you want traffic to
shed to lower-latency / cheaper healthy models before absolute p95 SLOs trip.

![Demo](../../assets/latency-slope-shed.gif)

## When to use it

- Provider latency is trending upward across recent samples.
- Absolute p95 shedding reacts too late for short spikes that start climbing.
- Healthy lower-latency / cheaper alternatives should absorb load first.

## How it works

1. Filter the catalog to domain-eligible models.
2. Pick the highest-quality candidate as the primary.
3. Compute EWMA slope (ms/step) over `NEXUS_LATENCY_SLOPE_WINDOW` samples.
4. If primary slope is at or under `NEXUS_LATENCY_SLOPE_THRESHOLD_MS`, keep it.
5. Otherwise shed to the lowest mean-latency / cheapest healthy (or stable)
   alternative.
6. The engine records latency samples into the shared slope window on success.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=latency-slope-shed
export NEXUS_LATENCY_SLOPE_WINDOW=10
export NEXUS_LATENCY_SLOPE_THRESHOLD_MS=25.0
```

Or per request:

```http
X-Router-Strategy: latency-slope-shed
```

## Tuning notes

- Raise the threshold to tolerate noisier latency while staying quality-first.
- Lower the window for faster reaction; raise it to smooth short blips.
- Slope windows are local to each router process unless shared externally.
