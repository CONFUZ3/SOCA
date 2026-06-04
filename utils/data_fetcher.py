"""Legacy import path for the data-fetching pipeline.

The implementation lives in ``utils/fetchers/`` as of the modular refactor.
This shim preserves backward compatibility so existing callers
(``from utils.data_fetcher import DataFetcher, DataFetchError, ...``)
continue to work unchanged.
"""

from __future__ import annotations

from utils.fetchers import (  # noqa: F401
    DataFetcher,
    DataFetchError,
    GeocodingError,
    PopulationDataError,
    NOMINATIM_URL,
    HDX_BASE_URL,
    PHOTON_URL,
    OVERTURE_CATEGORIES,
    _USER_AGENT,
    _MAX_RETRIES,
)

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
