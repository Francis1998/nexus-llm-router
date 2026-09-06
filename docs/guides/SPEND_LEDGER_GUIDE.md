# Spend Ledger Guide

![Spend ledger demo](../../assets/demo/spend-ledger.gif)

Durable SQLite spend tracking with `GET /v1/spend` aggregation. Closes the gap
versus LiteLLM virtual-key `/spend` APIs for GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 fleets.

## Why

In-memory budget counters reset on process restart. Portfolio demos and
multi-tenant gateways need a durable spend log that survives new process
instances pointing at the same database path.

## How it works

1. On each successful routed completion, `NexusRouter` appends a row to
   `SpendLedger` (SQLite).
2. Rows capture `request_id`, tenant (`user_id`), provider, model, cost, and
   token counts.
3. `GET /v1/spend` returns totals plus breakdowns by tenant / provider / model.
4. `POST /v1/spend` accepts manual records for demos or offline ingestion.

## Configuration

```bash
NEXUS_SPEND_LEDGER_PATH=migrations/spend-ledger.sqlite3
```

## Example

```bash
curl -s http://localhost:8000/v1/spend | jq
curl -s -X POST http://localhost:8000/v1/spend \
  -H 'content-type: application/json' \
  -d '{
    "request_id": "demo-1",
    "tenant": "acme",
    "provider": "openai",
    "model": "gpt-5.5",
    "cost_usd": 0.12
  }'
```

## Gap vs LiteLLM / Portkey

| Capability | LiteLLM virtual keys | Nexus spend ledger |
| --- | --- | --- |
| Durable spend log | Yes | Yes (SQLite) |
| `/spend` summary API | Yes | `GET /v1/spend` |
| Record on success | Yes | Wired in router `_respond` |
| Manual record API | Yes | `POST /v1/spend` |
