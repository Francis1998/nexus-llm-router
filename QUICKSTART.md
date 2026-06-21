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

## 4. Run the Offline Demo

The demo uses mock providers and prints model choice plus rationale traces.

```bash
PYTHONPATH=src python scripts/benchmark.py
```

Record a terminal demo GIF with your preferred terminal recorder:

```bash
PYTHONPATH=src python scripts/benchmark.py
```
