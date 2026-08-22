# Request-Class-QoS Routing Guide

Use `request-class-qos` when callers declare a QoS class and need differentiated
routing for interactive, batch, and bulk work across GPT-5.5 / Claude Sonnet
4.6 / Gemini 3.x / Kimi K2.

## When to use it

- Interactive chat and copilots need the lowest observed healthy latency with
  strong quality.
- Overnight or queue batch jobs should stay quality-first while preferring
  mid-cost models.
- Bulk ingestion or sweep traffic should minimize spend on healthy providers.

## How it works

1. Read `metadata.request_class` or `metadata.qos_class`.
2. Accept `interactive`, `batch`, or `bulk`; default and unknown values use
   interactive.
3. Filter domain-eligible candidates through provider circuit health.
4. Apply the class policy:
   - interactive: lowest observed provider p95, then quality
   - batch: highest quality with mid-cost preference
   - bulk: cheapest healthy model
5. If every provider is unhealthy, retain an emergency route from the eligible
   catalog so the request still decides.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=request-class-qos
```

Or select it per request:

```http
X-Router-Strategy: request-class-qos
```

```json
{"metadata": {"request_class": "interactive"}}
```

```json
{"metadata": {"qos_class": "bulk"}}
```
