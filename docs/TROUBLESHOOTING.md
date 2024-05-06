# Troubleshooting Router

*nexus-llm-router — 2024-05-06*

## Overview

This guide covers troubleshooting router for the `nexus-llm-router` project.

## Prerequisites

- Python 3.10+
- Redis (if using distributed mode)
- Environment variables configured (see `.env.example`)

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env

# Run the router module
python -m router --help
```

## Common Scenarios

### Scenario 1: Basic Router Usage

```python
from router import Router

client = Router(config)
result = client.run()
print(result)
```

### Scenario 2: Advanced Configuration

```python
from router.config import Settings

settings = Settings(
    max_retries=3,
    timeout=30,
    log_level="INFO",
)
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ConnectionError` | API endpoint unreachable | Check `BASE_URL` in `.env` |
| `TimeoutError` | Request took too long | Increase `timeout` setting |
| `AuthError` | Invalid or expired token | Rotate API key |

## See Also

- [README](../README.md)
- [ARCHITECTURE](../ARCHITECTURE.md)
- [API Reference](./API.md)
