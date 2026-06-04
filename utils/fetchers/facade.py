"""DataFetcher facade — preserves the legacy public API.

Callers import ``DataFetcher`` and call ``fetch_boundaries`` / ``fetch_pois``
/ ``fetch_population`` with the same signatures they used before the refactor.
Internally the facade delegates to module-level tier runners.
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd

from . import boundaries as _bnd
from . import pois as _pois
from . import population as _pop


class DataFetcher:
    """High-level interface for automatic geographic data retrieval.

    Usage::

        fetcher = DataFetcher()
        boundary   = fetcher.fetch_boundaries("Lima, Peru")
        population = fetcher.fetch_population(boundary)
        hospitals  = fetcher.fetch_pois(boundary, "health")
    """

    # No per-instance state needed — the shared Session + limiter live in
    # utils.fetchers.http. Kept as a class so legacy callers that instantiate
    # it continue to work.

    def fetch_boundaries(
        self,
        location: str,
        admin_level: Optional[int] = None,
        scale: str = "city",
        *,
        hint: Optional[object] = None,
        prefer_polygon: bool = True,  # kept for API compat; unused
    ) -> gpd.GeoDataFrame:
        return _bnd.fetch_boundaries(
            location, admin_level=admin_level, scale=scale, hint=hint,
        )

    def fetch_pois(
        self,
        boundary_gdf: gpd.GeoDataFrame,
        category: str,
    ) -> gpd.GeoDataFrame:
        return _pois.fetch_pois(boundary_gdf, category)

    def fetch_population(
        self,
        boundary_gdf: gpd.GeoDataFrame,
        n_points: Optional[int] = None,
        random_seed: int = 42,
    ) -> gpd.GeoDataFrame:
        return _pop.fetch_population(
            boundary_gdf, n_points=n_points, random_seed=random_seed,
        )
