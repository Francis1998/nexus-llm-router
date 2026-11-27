"""Suite-wide test configuration."""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo

import pytest

import router.config as config

# List prices are dated (``router.config.PRICE_SCHEDULE``), so price assertions
# would change on their own when a scheduled change takes effect. Every test is
# priced at this instant unless it pins its own.
SUITE_PRICING_INSTANT = datetime(2026, 11, 27, tzinfo=UTC)


class _SuiteDatetime(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:  # type: ignore[override]
        return SUITE_PRICING_INSTANT if tz is None else SUITE_PRICING_INSTANT.astimezone(tz)


@pytest.fixture(autouse=True)
def _pin_pricing_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the default pricing clock (``router.config.utc_now``) for each test."""
    monkeypatch.setattr(config, "datetime", _SuiteDatetime)
