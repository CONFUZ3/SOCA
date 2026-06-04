"""DuckDB-backed Overture Maps query layer.

Uses the public ``s3://overturemaps-us-west-2/release/<RELEASE>/...`` parquet
with SQL predicate pushdown, so queries filter at the parquet row-group
level rather than downloading and filtering in-memory. This is the path
recommended by the Overture docs:
    https://docs.overturemaps.org/getting-data/cloud-sources/
    https://docs.overturemaps.org/guides/divisions/
    https://docs.overturemaps.org/guides/places/

Falls back to the ``overturemaps`` Python client when DuckDB is missing.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from functools import lru_cache
from typing import Any, Optional

from .constants import (
    _OVERTURE_DEFAULT_RELEASE,
    _OVERTURE_READ_TIMEOUT_SEC,
    _OVERTURE_S3_BASE,
)
from .errors import DataFetchError
from .overture_release import get_overture_release

logger = logging.getLogger(__name__)

try:
    import duckdb  # type: ignore
    _DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore
    _DUCKDB_AVAILABLE = False


def is_available() -> bool:
    """True when the DuckDB path can be used."""
    return _DUCKDB_AVAILABLE


_conn_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_connection():
    """Lazy, process-wide DuckDB connection with spatial + httpfs loaded."""
    if not _DUCKDB_AVAILABLE:
        raise DataFetchError("duckdb not installed; run `pip install duckdb`.")
    conn = duckdb.connect(database=":memory:")
    conn.execute("INSTALL spatial;")
    conn.execute("LOAD spatial;")
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")
    conn.execute("SET s3_region='us-west-2';")
    return conn


def _theme_path(theme: str, type_: str) -> str:
    """Build the S3 parquet glob for a theme/type at the current release."""
    try:
        release = get_overture_release()
    except Exception:
        release = _OVERTURE_DEFAULT_RELEASE
    return f"{_OVERTURE_S3_BASE}/{release}/theme={theme}/type={type_}/*"


def _run_query(sql: str, params: list[Any]):
    """Run a query with a wall-clock timeout. Returns a pandas DataFrame."""
    conn = _get_connection()

    def _exec():
        # DuckDB connections aren't thread-safe; lock for cross-thread use.
        with _conn_lock:
            return conn.execute(sql, params).df()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_exec)
        try:
            return fut.result(timeout=_OVERTURE_READ_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError as exc:
            raise DataFetchError(
                f"Overture/DuckDB query exceeded {_OVERTURE_READ_TIMEOUT_SEC}s"
            ) from exc


def query_divisions(
    *,
    bbox: tuple[float, float, float, float],
    subtypes: list[str],
    name_query: str,
    country: Optional[str] = None,
):
    """Fetch `division` point rows that roughly match *name_query* in *bbox*."""
    xmin, ymin, xmax, ymax = bbox
    # Row bbox intersects query bbox iff it is NOT disjoint.
    where = [
        "bbox.xmax >= ?", "bbox.xmin <= ?",
        "bbox.ymax >= ?", "bbox.ymin <= ?",
        f"subtype IN ({','.join(['?'] * len(subtypes))})",
        "lower(names.primary) LIKE ?",
    ]
    params: list[Any] = [
        xmin, xmax, ymin, ymax,
        *subtypes,
        f"%{name_query.strip().lower()}%",
    ]
    if country:
        where.append("country = ?")
        params.append(country.upper())

    sql = f"""
        SELECT id,
               subtype,
               names.primary       AS name,
               country,
               population
        FROM read_parquet('{_theme_path("divisions", "division")}')
        WHERE {' AND '.join(where)}
        LIMIT 500
    """
    return _run_query(sql, params)


def query_division_area_by_id(
    division_id: str,
    *,
    bbox: Optional[tuple[float, float, float, float]] = None,
):
    """Fetch the polygon (as WKB bytes) for a given division id.

    When *bbox* is supplied it enables parquet row-group pruning on the
    bbox column, which is orders-of-magnitude faster than a global scan.
    """
    where = ["division_id = ?"]
    params: list[Any] = [division_id]
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        where = [
            "bbox.xmax >= ?", "bbox.xmin <= ?",
            "bbox.ymax >= ?", "bbox.ymin <= ?",
            *where,
        ]
        params = [xmin, xmax, ymin, ymax, *params]

    sql = f"""
        SELECT geometry
        FROM read_parquet('{_theme_path("divisions", "division_area")}')
        WHERE {' AND '.join(where)}
        LIMIT 1
    """
    return _run_query(sql, params)


def query_places(
    *,
    bbox: tuple[float, float, float, float],
    overture_categories: list[str],
):
    """Fetch `place` rows whose primary category is in *overture_categories*."""
    if not overture_categories:
        return None
    xmin, ymin, xmax, ymax = bbox
    where = [
        "bbox.xmax >= ?", "bbox.xmin <= ?",
        "bbox.ymax >= ?", "bbox.ymin <= ?",
        f"categories.primary IN ({','.join(['?'] * len(overture_categories))})",
    ]
    params: list[Any] = [
        xmin, xmax, ymin, ymax,
        *overture_categories,
    ]
    sql = f"""
        SELECT names.primary        AS name,
               categories.primary   AS amenity,
               geometry
        FROM read_parquet('{_theme_path("places", "place")}')
        WHERE {' AND '.join(where)}
        LIMIT 20000
    """
    return _run_query(sql, params)
