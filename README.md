# nexus-llm-router

![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![CI](https://github.com/Francis1998/{repo}/actions/workflows/ci.yml/badge.svg)

> Llm Routing — powered by modern Python async architecture.

## Features

- **Router engine** with configurable strategies
- **Adapter pipeline** with full observability
- **Async-first** design using `asyncio` + `aiohttp`
- **Type-safe** with full `mypy` compliance
- **Production-ready** with Docker, CI/CD, and structured logging

## Quick Start

```bash
git clone https://github.com/Francis1998/nexus-llm-router.git
cd nexus-llm-router
pip install -e ".[dev]"
cp .env.example .env
python -m router --help
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](ARCHITECTURE.md) | System design and component overview |
| [Configuration](docs/CONFIGURATION.md) | All configuration options |
| [Deployment](docs/DEPLOYMENT.md) | Production deployment guide |
| [Contributing](CONTRIBUTING.md) | Development and PR workflow |
| [Changelog](CHANGELOG.md) | Version history |

## License

MIT © [Francis1998](https://github.com/Francis1998)

*Last updated: 2024-10-30*
