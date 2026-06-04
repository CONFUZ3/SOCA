"""Lightweight data-source plugin registry.

Plugins are pre-coded adapters for specific data sources. The LLM picks one by
name and supplies parameters; it never writes retrieval code. Existing fetchers
(boundaries, pois, population) are unchanged — this registry is purely additive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import geopandas as gpd


class DataSourcePlugin(ABC):
    name: str
    description: str
    supported_categories: list[str]

    @abstractmethod
    def fetch(self, boundary_gdf: gpd.GeoDataFrame, **kwargs: Any) -> gpd.GeoDataFrame:
        ...

    @abstractmethod
    def validate_params(self, **kwargs: Any) -> tuple[bool, str]:
        ...


class SourceRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, DataSourcePlugin] = {}

    def register(self, plugin: DataSourcePlugin) -> None:
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> DataSourcePlugin | None:
        return self._plugins.get(name)

    def list_available(self) -> list[dict]:
        return [
            {
                "name": p.name,
                "description": p.description,
                "categories": p.supported_categories,
            }
            for p in self._plugins.values()
        ]


source_registry = SourceRegistry()
