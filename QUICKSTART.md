# Quickstart

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[dev]"
cp .env.example .env
```

## 2. Run Quality Gates

```bash
ruff check src/
mypy src/
pytest tests/ -v
```

## 3. Run the API

```bash
PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Health:

```bash
curl http://localhost:8000/health
```

Chat completion:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -H "X-Router-Strategy: classifier" \
  -d '{"messages":[{"role":"user","content":"Design a robust async retry policy."}],"max_tokens":128,"user":"demo"}'
```

OpenAI Responses API (works with `client.responses.create(...)` in the OpenAI SDKs):

```bash
curl -s http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "X-Router-Strategy: cost-optimal" \
  -d '{"instructions":"Be terse.","input":"Summarize our refund policy.","max_output_tokens":128}'
```

Anthropic Messages API (works with `anthropic.Anthropic(base_url="http://localhost:8000")`; text only):

```bash
curl -s http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: <your-nexus-key>" \
  -d '{"model":"claude-haiku-5-5","max_tokens":128,"system":"Be terse.","messages":[{"role":"user","content":"Summarize our refund policy."}]}'
```

No provider keys yet? Serve the same API with deterministic mock providers:

```bash
PYTHONPATH=src python scripts/serve_mock.py --port 8000
```

See [docs/guides/RESPONSES_API_GUIDE.md](docs/guides/RESPONSES_API_GUIDE.md) and
[docs/guides/ANTHROPIC_MESSAGES_API_GUIDE.md](docs/guides/ANTHROPIC_MESSAGES_API_GUIDE.md)
for streaming and supported fields.

Rate an answer, then let feedback steer routing (`router.request_id` comes from
the chat completion; send the same API key and `X-Tenant-Id` as that request):

```bash
curl -s http://localhost:8000/v1/feedback \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{"request_id":"<router.request_id>","value":true,"comment":"Clear and correct."}'
curl -s "http://localhost:8000/v1/feedback/summary?group_by=model" -H "Authorization: Bearer local-key"
```

Send `X-Router-Strategy: feedback-aware` to route by that feedback once enough
has been collected; see [docs/guides/FEEDBACK_LOOP_GUIDE.md](docs/guides/FEEDBACK_LOOP_GUIDE.md).

## 4. Run the Offline Demo

The demo uses mock providers and prints model choice plus rationale traces.

```bash
PYTHONPATH=src python scripts/benchmark.py
```

Record a terminal demo GIF with your preferred terminal recorder:

```bash
PYTHONPATH=src python scripts/benchmark.py
```
