"""Exception types for the fetchers package."""

from __future__ import annotations


class DataFetchError(Exception):
    """Base exception for all data-fetching failures."""


class GeocodingError(DataFetchError):
    """Raised when boundary geocoding fails across all tiers."""


class PopulationDataError(DataFetchError):
    """Raised when a valid population grid cannot be produced."""
