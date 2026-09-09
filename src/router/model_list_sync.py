"""Offline provider model-list sync into the capability catalog."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from router.capability_catalog import ModelCapabilityCatalog


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Static / provider-supplied model capability descriptor.

    Used by offline sync jobs that close the LiteLLM ``model_list`` /
    Portkey catalog-refresh gap without contacting the network.
    """

    model_id: str
    capabilities: frozenset[str] | set[str] | tuple[str, ...] | list[str]
    provider: str | None = None

    def __post_init__(self) -> None:
        """Normalize capabilities to a frozenset and validate ``model_id``."""
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))


@dataclass(frozen=True, slots=True)
class ModelListSyncResult:
    """Outcome of a deterministic model-list sync pass."""

    refreshed_from_static: int
    upserted: int
    model_ids: tuple[str, ...]


class ProviderModelListSync:
    """Sync static or provided model lists into ``ModelCapabilityCatalog``.

    LiteLLM refreshes a live ``model_list`` and Portkey refreshes a hosted
    model catalog. Nexus keeps routing offline-first: this helper seeds via
    ``refresh_from_static`` and applies deterministic ``upsert`` calls from a
    provided descriptor list — covering GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 without network I/O.
    """

    def __init__(self, catalog: ModelCapabilityCatalog | None = None) -> None:
        """Attach to an existing catalog or create an empty one.

        Args:
            catalog: Target capability catalog. When omitted, starts empty
                (``auto_seed=False``) so callers control the first sync.
        """
        self._catalog = catalog if catalog is not None else ModelCapabilityCatalog(auto_seed=False)

    @property
    def catalog(self) -> ModelCapabilityCatalog:
        """Return the target capability catalog."""
        return self._catalog

    def sync_static(self) -> ModelListSyncResult:
        """Replace the catalog with the built-in static known-model seed.

        Returns:
            Sync result with ``refreshed_from_static`` set to the seed size.
        """
        refreshed = self._catalog.refresh_from_static()
        return ModelListSyncResult(
            refreshed_from_static=refreshed,
            upserted=0,
            model_ids=tuple(sorted(self._catalog.snapshot())),
        )

    def sync(
        self,
        descriptors: Sequence[ModelDescriptor] | Iterable[ModelDescriptor],
        *,
        refresh_static_first: bool = False,
    ) -> ModelListSyncResult:
        """Upsert descriptors into the catalog (optionally after a static refresh).

        Args:
            descriptors: Offline model descriptors to apply in order.
            refresh_static_first: When ``True``, call ``refresh_from_static``
                before upserting so the seed map is restored first.

        Returns:
            Sync result summarizing refresh + upsert counts and final model ids.
        """
        refreshed = 0
        if refresh_static_first:
            refreshed = self._catalog.refresh_from_static()
        upserted = 0
        for descriptor in descriptors:
            if not isinstance(descriptor, ModelDescriptor):
                raise TypeError("descriptors must be ModelDescriptor instances")
            self._catalog.upsert(descriptor.model_id, descriptor.capabilities)
            upserted += 1
        return ModelListSyncResult(
            refreshed_from_static=refreshed,
            upserted=upserted,
            model_ids=tuple(sorted(self._catalog.snapshot())),
        )
