# Contributing to nexus-llm-router

Thank you for considering a contribution.

## Development Setup

```bash
git clone https://github.com/Francis1998/nexus-llm-router.git
cd nexus-llm-router
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

## Quality Gates

Run these before opening a pull request:

```bash
ruff check src/ tests/ scripts/
mypy src/
pytest tests/ -v
```

## Coding Standards

- Python 3.11+
- Type annotations on all functions
- Google-style docstrings
- Ruff for linting (`ruff check src/ tests/ scripts/`)

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Ensure the quality gates above pass
3. Update relevant documentation
4. Open a PR with a clear description of the change

## Adding an Advisor

Offline advisor and guard modules live in `src/safety/`, each with a guide in
`docs/guides/` and tests in `tests/`.

- Add the advisor's catalog entry (`### <Name>Advisor` heading, demo GIF, and a
  one-paragraph summary) to [docs/ADVISOR_CATALOG.md](docs/ADVISOR_CATALOG.md),
  plus a row in its index table. Do not add advisor entries to `README.md`; the
  README stays focused on the product overview.
- Reference GIFs from `docs/` as `../assets/demo/<name>.gif`.
- Describe the current model stack from the README's
  [Current Model Stack](README.md#current-model-stack) table instead of
  hard-coding older model names.

## Commit Message Format

```text
<type>(<scope>): <short summary>

type: feat | fix | docs | refactor | test | chore
```
