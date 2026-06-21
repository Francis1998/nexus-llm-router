# Nexus LLM Router

Intelligent multi-LLM router with task-aware routing strategies, cost optimization, and production safety controls. Nexus exposes a drop-in OpenAI-compatible `/v1/chat/completions` API and routes each request through an opinionated `Observe -> Decide -> Act` pipeline.

## Why Nexus

Nexus is designed for AI infrastructure engineers running multi-model production pipelines where quality, latency, and cost all matter. It is inspired by the problem space explored by LiteLLM, RouteLLM, and anyscale/llm-router, but uses an original architecture centered on deterministic routing decisions, durable audit logs, and explicit safety controls.

## Core Features

- `Observe`: extracts complexity score, domain tag, latency requirement, and token budget.
- `Decide`: selects models with rule-based, classifier, cost-optimal, latency-aware, or A/B strategies.
- `Act`: dispatches through provider adapters and automatically walks fallback chains.
- State machine: `RECEIVED -> CLASSIFIED -> ROUTED -> DISPATCHED -> RESPONDED | FALLBACK | FAILED`.
- Provider adapters: OpenAI, Anthropic, Google Gemini, and Moonshot Kimi.
- Safety: circuit breakers, request timeouts, budget caps, PII scrubbing, and token-bucket rate limits.
- Observability: structlog JSON logs, OpenTelemetry traces, Prometheus metrics, and Grafana dashboard.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[dev]"
cp .env.example .env
PYTHONPATH=src uvicorn api.main:app --reload
```

Send a request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Router-Strategy: rule-based" \
  -d '{"messages":[{"role":"user","content":"Debug this Python retry loop."}],"max_tokens":128}'
```

Run the offline routing demo:

```bash
PYTHONPATH=src python scripts/benchmark.py
```

## Quality Gates

```bash
ruff check src/
mypy src/
pytest tests/ -v
```

## Docker Compose

```bash
cp .env.example .env
docker-compose up --build
```

Services:

- Router: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Routing Strategies

Select a strategy with `X-Router-Strategy`:

- `rule-based`: domain and complexity priority matrix.
- `classifier`: logistic-regression-style complexity and domain features.
- `cost-optimal`: minimizes estimated cost subject to quality floor.
- `latency-aware`: penalizes providers with poor rolling p95 latency.
- `ab`: deterministic request-id buckets across two model arms.

## Documentation

- [QUICKSTART.md](QUICKSTART.md)
- [CONFIGURATION.md](CONFIGURATION.md)
- [SAFETY.md](SAFETY.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)

## License

Apache-2.0
