"""POI fetchers: Overture Maps only.

Degenerate geometries (empty, zero-area) are filtered explicitly with a
count log rather than being centroided silently, so the user can see when
data quality is poor.
"""

from __future__ import annotations

import concurrent.futures
import logging

import geopandas as gpd
import overturemaps  # type: ignore
import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
from shapely.ops import unary_union

from .constants import (
    OVERTURE_CATEGORIES,
    _OVERTURE_READ_TIMEOUT_SEC,
)
from .errors import DataFetchError

logger = logging.getLogger(__name__)


def _clip_to_boundary(gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        boundary_union = unary_union(
            boundary_gdf.to_crs("EPSG:4326").geometry.values
        )
        clipped = gdf[gdf.geometry.within(boundary_union)].copy()
        logger.info(f"POI clip: {len(gdf)} → {len(clipped)} inside boundary")
        return clipped.reset_index(drop=True)
    except Exception as clip_err:
        logger.warning(f"Could not clip POIs to boundary: {clip_err}")
        return gdf


def _reduce_to_points(geoms):
    """Convert non-Point geometries to centroids, dropping empty/zero-area.

    Returns (list_of_points, stats_dict) where stats_dict reports how many
    geometries were dropped so the caller can log data-quality info.
    """
    points = []
    kept_point = 0
    centroided = 0
    dropped_empty = 0
    dropped_zero_area = 0
    for g in geoms:
        if g is None or g.is_empty:
            dropped_empty += 1
            points.append(None)
            continue
        if g.geom_type == "Point":
            points.append(g)
            kept_point += 1
            continue
        # For polygons, reject zero-area (degenerate) geometries outright.
        if g.geom_type in ("Polygon", "MultiPolygon") and g.area == 0:
            dropped_zero_area += 1
            points.append(None)
            continue
        c = g.centroid
        if c is None or c.is_empty:
            dropped_empty += 1
            points.append(None)
            continue
        points.append(c)
        centroided += 1
    stats = {
        "kept_point": kept_point,
        "centroided": centroided,
        "dropped_empty": dropped_empty,
        "dropped_zero_area": dropped_zero_area,
    }
    return points, stats


def _assemble_pois_gdf(df, category: str) -> gpd.GeoDataFrame:
    """Shared WKB → points → GeoDataFrame tail. Accepts a DataFrame with
    columns: name, amenity, geometry (WKB bytes)."""
    import shapely.wkb as wkb
    empty = gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")
    if df is None or df.empty:
        return empty

    geoms = [wkb.loads(bytes(g)) for g in df["geometry"]]
    points, stats = _reduce_to_points(geoms)
    df = df.assign(_pt=points)
    df = df[df["_pt"].notna()]
    if stats["centroided"] or stats["dropped_empty"] or stats["dropped_zero_area"]:
        logger.info(f"Overture POIs reduction: {stats}")
    if df.empty:
        return empty
    df["name"] = df["name"].fillna("").astype(str)
    df["amenity"] = df["amenity"].fillna(category).astype(str)
    return gpd.GeoDataFrame(
        df[["name", "amenity"]].reset_index(drop=True),
        geometry=list(df["_pt"]),
        crs="EPSG:4326",
    )


def _fetch_pois_via_duckdb(bbox: tuple, category: str) -> gpd.GeoDataFrame:
    from . import overture_duckdb as od
    overture_cats = OVERTURE_CATEGORIES.get(category, [])
    empty = gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")
    if not overture_cats:
        return empty
    df = od.query_places(bbox=bbox, overture_categories=overture_cats)
    return _assemble_pois_gdf(df, category)


def _fetch_pois_via_pyclient(bbox: tuple, category: str) -> gpd.GeoDataFrame:
    overture_cats = OVERTURE_CATEGORIES.get(category, [])
    empty = gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")
    if not overture_cats:
        return empty

    try:
        reader = overturemaps.record_batch_reader("place", bbox=bbox)
        if reader is None:
            return empty
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            _f = _pool.submit(reader.read_all)
            try:
                table = _f.result(timeout=_OVERTURE_READ_TIMEOUT_SEC)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    f"Overture place read_all timed out after {_OVERTURE_READ_TIMEOUT_SEC}s"
                )
                return empty
    except Exception as exc:
        logger.warning(f"Overture reader error: {exc}")
        return empty

    if table.num_rows == 0:
        return empty

    try:
        primary_categories = pc.struct_field(table.column("categories"), "primary")
        mask = pc.is_in(primary_categories, value_set=pa.array(overture_cats))
        filtered_table = table.filter(mask)
    except Exception as filter_exc:
        logger.warning(f"Overture pyarrow filter failed ({filter_exc}); pandas fallback.")
        pdf = table.to_pandas()
        pdf["primary_cat"] = pdf["categories"].apply(
            lambda x: x.get("primary") if isinstance(x, dict) else None
        )
        pdf = pdf[pdf["primary_cat"].isin(overture_cats)]
        filtered_table = pa.Table.from_pandas(pdf)

    if filtered_table.num_rows == 0:
        return empty

    df = filtered_table.to_pandas()
    df["name"] = df["names"].apply(
        lambda x: x.get("primary", "") if isinstance(x, dict) else ""
    )
    df["amenity"] = df["categories"].apply(
        lambda x: x.get("primary", category) if isinstance(x, dict) else category
    )
    return _assemble_pois_gdf(df[["name", "amenity", "geometry"]], category)


def fetch_pois_via_overture(bbox: tuple, category: str) -> gpd.GeoDataFrame:
    """Fetch POIs from the Overture Maps 'place' theme.

    Primary path is DuckDB SQL (predicate pushdown on `categories.primary`);
    falls back to the `overturemaps` Python client when DuckDB is missing.
    """
    from . import overture_duckdb as od
    if od.is_available():
        return _fetch_pois_via_duckdb(bbox, category)
    logger.warning(
        "duckdb not installed — falling back to slow overturemaps client for POIs."
    )
    return _fetch_pois_via_pyclient(bbox, category)


def fetch_pois(
    boundary_gdf: gpd.GeoDataFrame,
    category: str,
) -> gpd.GeoDataFrame:
    """Fetch POIs from Overture Maps for the given category."""
    if category not in OVERTURE_CATEGORIES:
        raise DataFetchError(
            f"Unknown POI category '{category}'. "
            f"Supported: {sorted(OVERTURE_CATEGORIES.keys())}"
        )

    bounds = boundary_gdf.to_crs("EPSG:4326").total_bounds
    bbox = (bounds[0], bounds[1], bounds[2], bounds[3])

    pois_gdf = fetch_pois_via_overture(bbox, category)
    if pois_gdf.empty:
        raise DataFetchError(
            f"No '{category}' POIs found via Overture Maps for this area. "
            "The category may not be well-represented in Overture for this region."
        )

    pois_gdf = _clip_to_boundary(pois_gdf, boundary_gdf)
    pois_gdf["data_source"] = "overture_pois"
    pois_gdf["candidates_are_synthetic"] = False
    logger.info(f"POI fetch: {len(pois_gdf)} '{category}' via Overture Maps.")
    return pois_gdf
