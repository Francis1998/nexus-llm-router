# Use Cases: nexus-llm-router End to End

Three scenarios showing what nexus changes for a product team, each with the
exact commands and the real output they produced. All three run against the
mock-provider server, so no API keys are needed. Mock providers answer
`[<model>] routed response`, while routing, budgets, rate limits, idempotency,
and the audit log run for real.

| Scenario | Problem | Nexus features used |
| --- | --- | --- |
| [1. Cut spend for a support assistant](#1-cut-spend-for-a-support-assistant) | every message hits the most expensive model | `cost-optimal` and `rule-based` routing, spend ledger, key fingerprints |
| [2. Move an OpenAI SDK app behind nexus](#2-move-an-openai-sdk-app-behind-nexus) | multi-tenant app needs fair limits and safe retries without rewriting SDK code | Responses API, `X-Tenant-Id` rate limits, `Idempotency-Key` replay |
| [3. Route regulated requests with an audit trail](#3-route-regulated-requests-with-an-audit-trail) | clinical questions must stay on the safety model, with proof afterwards | domain-aware routing, audit log, PII scrubbing |

## Setup

```bash
git clone https://github.com/Francis1998/nexus-llm-router.git && cd nexus-llm-router
pip install -e ".[dev]"
NEXUS_TENANT_RATE_LIMIT_CAPACITY=3 NEXUS_TENANT_RATE_LIMIT_REFILL_PER_SECOND=0.01 \
  PYTHONPATH=src python scripts/serve_mock.py --port 8000
```

The two `NEXUS_TENANT_RATE_LIMIT_*` settings shrink the per-tenant bucket so
scenario 2 reaches a 429 within a few requests (defaults: 60 tokens, refilled at
1 per second). The commands use `jq` to trim responses.

## 1. Cut spend for a support assistant

**Situation.** A support assistant sends every message to one frontier model,
from "what are your hours?" to a production debugging session.

**With nexus.** Simple questions go to an economy model, hard code problems go
to the frontier model, and spend is attributed to the calling API key.

A FAQ-style question, routed with `X-Router-Strategy: cost-optimal`:

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -H 'Authorization: Bearer sk-demo-support' -H 'X-Router-Strategy: cost-optimal' \
  -d '{"messages":[{"role":"user","content":"What are your support hours?"}]}' \
  | jq '{model, strategy: .router.routing_strategy, rationale: .router.rationale, cost_usd: .router.cost_usd}'
```

```json
{
  "model": "gpt-6-luna",
  "strategy": "cost-optimal",
  "rationale": "LP objective minimized estimated cost $0.000257 with quality floor 0.72",
  "cost_usd": 0.000023
}
```

A debugging request with a code block (`hard-prompt.json`), sent without a
strategy header so the default `rule-based` policy decides:

````bash
cat > hard-prompt.json <<'JSON'
{"messages": [{"role": "user", "content": "Debug this Python function and refactor it for concurrency:\n```python\ndef charge(order):\n    for attempt in range(3):\n        try:\n            return gateway.charge(order.id, order.amount)\n        except TimeoutError:\n            time.sleep(2 ** attempt)\n    raise RuntimeError('payment failed')\n```\nRewrite it with asyncio, idempotency keys and jittered exponential backoff, then analyze the race conditions, prove the retry bound, and compare the trade-offs step by step."}]}
JSON
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -H 'Authorization: Bearer sk-demo-support' -d @hard-prompt.json \
  | jq '{model, strategy: .router.routing_strategy, rationale: .router.rationale, cost_usd: .router.cost_usd}'
````

```json
{
  "model": "gpt-6-astra",
  "strategy": "rule-based",
  "rationale": "complex code prompt favors GPT-6 Astra quality",
  "cost_usd": 0.000134
}
```

Where the money went:

```bash
curl -s localhost:8000/v1/spend | jq
```

```json
{
  "total_cost_usd": 0.000157,
  "request_count": 2,
  "by_tenant": {
    "key_15b184c4c9b35a2f": 0.000157
  },
  "by_provider": {
    "openai": 0.000157
  },
  "by_model": {
    "gpt-6-astra": 0.000134,
    "gpt-6-luna": 0.000023
  }
}
```

What to notice:

- `cost-optimal` chose `gpt-6-luna`, the cheapest catalog model that clears the
  quality floor; its decision-time estimate uses the catalog list prices.
- The default `rule-based` policy recognized a complex code prompt (code block,
  debugging, step-by-step analysis) and chose `gpt-6-astra`.
- Spend is attributed to `key_15b184c4c9b35a2f`, a non-reversible fingerprint of
  the API key; the raw `sk-demo-support` key is never stored or echoed.
- Recorded `cost_usd` comes from the provider adapter. The mock adapter bills a
  flat $0.001 per 1K tokens; real adapters bill the list prices in the README's
  [Current Model Stack](../../README.md#current-model-stack) table, including
  long-prompt tiers. `NEXUS_BUDGET_CAP_USD` (default `25`) rejects a request
  before dispatch when its estimate would push a user or session over the cap.

## 2. Move an OpenAI SDK app behind nexus

**Situation.** A product already uses the official OpenAI SDK and the Responses
API. It serves several customers (tenants) and retries failed calls, and a retry
must not run or bill a request twice.

**With nexus.** Change `base_url` and keep the SDK. Add `X-Tenant-Id` for
per-tenant rate limits and `Idempotency-Key` for safe retries.

`usecase_sdk.py`:

```python
"""Scenario 2: an OpenAI SDK app on nexus with tenant guards."""

import openai
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-demo-acme", max_retries=0)
headers = {"X-Tenant-Id": "acme", "Idempotency-Key": "release-note-7"}
prompt = "Draft a two-line release note for v2.4."

first = client.responses.create(model="gpt-6-luna", input=prompt, extra_headers=headers)
retry = client.responses.create(model="gpt-6-luna", input=prompt, extra_headers=headers)
nexus = (first.model_extra or {}).get("nexus", {})
print("first :", first.id, first.model, repr(first.output_text))
print("retry :", retry.id, "(replayed)" if retry.id == first.id else "(new response)")
print("nexus :", nexus.get("routing_strategy"), "|", nexus.get("rationale"))

for attempt in range(1, 4):
    try:
        client.responses.create(
            model="gpt-6-luna", input=f"Ping {attempt}", extra_headers={"X-Tenant-Id": "acme"}
        )
        print(f"ping {attempt}: ok")
    except openai.RateLimitError as exc:
        print(f"ping {attempt}: HTTP {exc.status_code} {exc.body}")
```

```bash
uv run --with openai python usecase_sdk.py   # or: pip install openai && python usecase_sdk.py
```

```text
first : resp_99efb5126ed6492ca05eeaaecc57bb92 gemini-3.8-flash '[gemini-3.8-flash] routed response'
retry : resp_99efb5126ed6492ca05eeaaecc57bb92 (replayed)
nexus : rule-based | simple realtime prompt favors low latency
ping 1: ok
ping 2: HTTP 429 {'message': 'tenant rate limit exceeded for acme', 'type': 'rate_limit_error', 'param': None, 'code': 'rate_limit_exceeded'}
ping 3: HTTP 429 {'message': 'tenant rate limit exceeded for acme', 'type': 'rate_limit_error', 'param': None, 'code': 'rate_limit_exceeded'}
```

What to notice:

- The retry with the same `Idempotency-Key` replayed the stored response (same
  `resp_...` id) instead of calling a provider again.
- `model="gpt-6-luna"` is a hint. For this short prompt the default `rule-based`
  strategy preferred `gemini-3.8-flash`, and `response.model` reports the model
  that actually served it. [Choosing a model](RESPONSES_API_GUIDE.md#choosing-a-model)
  explains when the requested model is honored; `X-Router-Strategy` selects the
  policy per request.
- The `acme` bucket held 3 tokens. The first request, its replay, and `ping 1`
  used them, so `ping 2` and `ping 3` got HTTP 429 with an OpenAI-style
  `rate_limit_error`, which the SDK raises as `openai.RateLimitError`. Other
  tenants have their own buckets.
- `max_retries=0` keeps the demo deterministic; with the SDK default, 429s are
  retried with backoff.

## 3. Route regulated requests with an audit trail

**Situation.** A health-tech team must keep clinical questions on the most
safety-focused model and prove afterwards which model answered and why.

**With nexus.** Domain-aware rules send medical and legal prompts to Claude
Opus 5.5, and every request is appended to a JSONL audit log.

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -H 'X-Router-Strategy: rule-based' \
  -d '{"messages":[{"role":"user","content":"A patient on warfarin asks whether ibuprofen is safe for back pain. Summarize the interaction risk for the care team."}]}' \
  | jq '{model, rationale: .router.rationale, request_id: .router.request_id}'
```

```json
{
  "model": "claude-opus-5-5",
  "rationale": "medical domain requires highest safety prior",
  "request_id": "efbda677902844abbb9e25996499b395"
}
```

```bash
tail -n 1 migrations/audit-log.jsonl | jq
```

```json
{
  "request_id": "efbda677902844abbb9e25996499b395",
  "chosen_model": "claude-opus-5-5",
  "provider": "anthropic",
  "routing_strategy": "rule-based",
  "rationale": "medical domain requires highest safety prior",
  "latency_ms": 0.1127077266573906,
  "token_cost": 0.000044999999999999996,
  "input_tokens": 29,
  "output_tokens": 16,
  "state": "RESPONDED"
}
```

What to notice:

- The prompt was classified as medical, and the rule-based policy routed it to
  `claude-opus-5-5`; legal prompts take the same path ("legal domain favors
  Claude policy reasoning").
- The audit record ties the `request_id` returned to the client to the model,
  provider, strategy, rationale, latency, tokens, cost, and final state. It
  does not store the prompt text.
- Set `NEXUS_ENABLE_PII_SCRUBBING=true` to redact email addresses and US phone
  numbers before dispatch ([SAFETY.md](../../SAFETY.md#pii-scrubbing)).

## How these transcripts were recorded

Recorded on 2026-11-07 from a real run of `scripts/serve_mock.py`. Port 8000 was
busy on the recording machine, so the server listened on port 8765, and
`NEXUS_AUDIT_LOG_PATH`, `NEXUS_SPEND_LEDGER_PATH`, and `NEXUS_IDEMPOTENCY_PATH`
pointed to a scratch directory so earlier runs could not leak into the output.
The commands above show the default port and paths. Request ids, response ids,
and latencies differ on every run.
