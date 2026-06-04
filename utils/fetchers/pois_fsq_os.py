"""Foursquare OS Places fetcher via Iceberg REST catalog.

Access requires a free account and access token from the Foursquare Places
Portal (https://location.foursquare.com/developer/).  Set the token in the
environment:

    FSQ_OS_ACCESS_TOKEN=<your-token>

If the token is absent or pyiceberg is not installed, every call raises
DataFetchError immediately so the caller can skip this tier gracefully.

Catalog details (fill in once your account is provisioned):
    FSQ_OS_CATALOG_URI  — REST catalog endpoint URL
    FSQ_OS_CATALOG_WAREHOUSE — catalog warehouse / namespace
    FSQ_OS_TABLE_NAME   — fully-qualified Iceberg table name
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import Optional

import geopandas as gpd
from shapely.geometry import Point

from .constants import (
    FSQ_OS_CATEGORIES,
    _FSQ_OS_READ_TIMEOUT_SEC,
)
from .errors import DataFetchError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog coordinates — update these once your FSQ portal account is active.
# ---------------------------------------------------------------------------
_FSQ_OS_CATALOG_URI = os.getenv(
    "FSQ_OS_CATALOG_URI",
    "https://api.foursquare.com/v3/places/iceberg",   # placeholder — verify in portal
)
_FSQ_OS_CATALOG_WAREHOUSE = os.getenv("FSQ_OS_CATALOG_WAREHOUSE", "foursquare")
_FSQ_OS_TABLE_NAME = os.getenv("FSQ_OS_TABLE_NAME", "foursquare.os_places")
_FSQ_OS_LIMIT = 20_000

try:
    from pyiceberg.catalog import load_catalog  # type: ignore
    from pyiceberg.expressions import (  # type: ignore
        And, GreaterThanOrEqual, LessThanOrEqual
    )
    _PYICEBERG_AVAILABLE = True
except ImportError:
    _PYICEBERG_AVAILABLE = False


def _get_catalog(access_token: str):
    """Return a pyiceberg REST catalog authenticated with *access_token*."""
    return load_catalog(
        "foursquare",
        **{
            "type": "rest",
            "uri": _FSQ_OS_CATALOG_URI,
            "warehouse": _FSQ_OS_CATALOG_WAREHOUSE,
            "token": access_token,
        },
    )


def _label_matches(label: str, targets: list[str]) -> bool:
    """True if *label* contains any of the *targets* (case-insensitive)."""
    lower = label.lower()
    return any(t.lower() in lower for t in targets)


def fetch_pois_via_fsq_os(
    bbox: tuple[float, float, float, float],
    category: str,
    access_token: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Fetch POIs from the Foursquare OS Places Iceberg catalog.

    Parameters
    ----------
    bbox:
        (xmin, ymin, xmax, ymax) in EPSG:4326.
    category:
        One of the 10 standard SOCA POI categories.
    access_token:
        FSQ portal token.  Falls back to the ``FSQ_OS_ACCESS_TOKEN`` env var.

    Returns
    -------
    GeoDataFrame with columns ["name", "amenity", "geometry"], CRS EPSG:4326.
    Returns an *empty* GeoDataFrame when no features match (does not raise).
    Raises DataFetchError on auth/config/transport failures.
    """
    if not _PYICEBERG_AVAILABLE:
        raise DataFetchError(
            "pyiceberg is not installed; run `pip install pyiceberg` to enable "
            "the Foursquare OS Places tier."
        )

    token = access_token or os.getenv("FSQ_OS_ACCESS_TOKEN", "")
    if not token:
        raise DataFetchError(
            "FSQ_OS_ACCESS_TOKEN is not set.  Create a free account at "
            "https://location.foursquare.com/developer/ and export the token."
        )

    targets = FSQ_OS_CATEGORIES.get(category, [])
    if not targets:
        raise DataFetchError(f"No FSQ OS category mapping for '{category}'.")

    empty = gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")

    xmin, ymin, xmax, ymax = bbox

    def _query() -> gpd.GeoDataFrame:
        catalog = _get_catalog(token)
        table = catalog.load_table(_FSQ_OS_TABLE_NAME)

        # Bbox predicate — pushes down to Iceberg file pruning.
        row_filter = And(
            And(
                GreaterThanOrEqual("longitude", xmin),
                LessThanOrEqual("longitude", xmax),
            ),
            And(
                GreaterThanOrEqual("latitude", ymin),
                LessThanOrEqual("latitude", ymax),
            ),
        )

        scan = table.scan(
            row_filter=row_filter,
            selected_fields=("name", "latitude", "longitude", "fsq_category_labels"),
            limit=_FSQ_OS_LIMIT,
        )
        df = scan.to_pandas()

        if df.empty:
            return empty

        # fsq_category_labels is a list<string> column; filter rows whose
        # label list contains at least one target substring.
        def _row_matches(labels) -> bool:
            if not labels:
                return False
            return any(_label_matches(lbl, targets) for lbl in labels)

        mask = df["fsq_category_labels"].apply(_row_matches)
        df = df[mask].copy()

        if df.empty:
            return empty

        df = df.dropna(subset=["latitude", "longitude"]).head(_FSQ_OS_LIMIT)
        geometries = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
        return gpd.GeoDataFrame(
            {
                "name": df["name"].fillna("").values,
                "amenity": category,
                "geometry": geometries,
            },
            crs="EPSG:4326",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_query)
        try:
            return fut.result(timeout=_FSQ_OS_READ_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError as exc:
            raise DataFetchError(
                f"FSQ OS Places Iceberg query exceeded {_FSQ_OS_READ_TIMEOUT_SEC}s"
            ) from exc
        except DataFetchError:
            raise
        except Exception as exc:
            raise DataFetchError(f"FSQ OS Places fetch failed: {exc}") from exc
