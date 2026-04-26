"""Data-fetching pipeline package.

Split out from the monolithic utils/data_fetcher.py for testability:
  - http         : shared Session + thread-safe Nominatim rate limiter
  - validation   : polygon + scale-match validation
  - boundaries   : Overture (primary) → Nominatim fallback
  - pois         : Overture Maps only
  - population   : Kontur (HDX) → synthetic; robust column resolver
  - facade       : DataFetcher orchestrator (same public API as before)

The legacy import path ``from utils.data_fetcher import DataFetcher, ...``
continues to work via a compat shim at utils/data_fetcher.py.
"""

from __future__ import annotations

from .errors import (
    DataFetchError,
    GeocodingError,
    PopulationDataError,
)
from .constants import (
    NOMINATIM_URL,
    HDX_BASE_URL,
    PHOTON_URL,
    OVERTURE_CATEGORIES,
    _USER_AGENT,
    _MAX_RETRIES,
)
from .facade import DataFetcher

__all__ = [
    "DataFetcher",
    "DataFetchError",
    "GeocodingError",
    "PopulationDataError",
    "NOMINATIM_URL",
    "HDX_BASE_URL",
    "PHOTON_URL",
    "OVERTURE_CATEGORIES",
    "_USER_AGENT",
    "_MAX_RETRIES",
]
