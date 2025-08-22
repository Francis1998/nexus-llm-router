# ADR-020: Database Orm for nexus-llm-router

**Date:** 2025-08-22
**Status:** Accepted
**Context:** Llm Routing

## Context

The `router` module needs a reliable database ORM solution
that integrates cleanly with our async router pipeline.

## Decision

Use **SQLAlchemy 2.0** for database ORM.

## Considered Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **SQLAlchemy 2.0** (chosen) | Native async, well-maintained | Slightly higher cold-start |
| Tortoise ORM | Mature ecosystem | Sync-first, harder to integrate |
| raw SQL | Zero dependencies | Limited features for production |

## Consequences

- All new router components will use `SQLAlchemy 2.0` as the database ORM layer.
- Existing code will be migrated incrementally.
- Added to `pyproject.toml` as a core dependency.
