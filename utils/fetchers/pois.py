"""POI fetchers: Overture Maps unioned with OSM Overpass.

Two tiers run in parallel and their results are spatially deduplicated so a
region where one provider is sparse (Overture in many parts of Africa /
South Asia; OSM in some private-data regions) still produces complete
facility coverage. Empty Overture is no longer fatal — fetch_pois only
raises when both tiers return zero useful features.

Degenerate geometries (empty, zero-area) are filtered explicitly with a
count log rather than being centroided silently, so the user can see when
data quality is poor.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
from typing import Optional

import geopandas as gpd
import overturemaps  # type: ignore
import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
from shapely.ops import unary_union

from .constants import (
    OSM_AMENITY_TAGS,
    OVERTURE_CATEGORIES,
    _OVERTURE_READ_TIMEOUT_SEC,
    _POI_DEDUP_RADIUS_M,
)
from .errors import DataFetchError
from .pois_overpass import fetch_pois_via_overpass

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
        # A failed clip is recoverable (return unclipped), but the user must
        # see it — silently returning unclipped data has hidden boundary bugs
        # before. Mirror to the activity log so the sidebar shows a warning.
        logger.warning(f"Could not clip POIs to boundary: {clip_err}")
        try:
            from utils.activity_log import log_event
            log_event(
                "fetch.pois",
                "info",
                f"POI clip to boundary failed ({clip_err}); "
                f"returning unclipped {len(gdf)} points",
                source="OpenStreetMap",
            )
        except Exception:  # pragma: no cover - activity_log is optional
            pass
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


# ---------------------------------------------------------------------------
# Dedup helpers for Overture ∪ Overpass union
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _name_tokens(name: Optional[str]) -> set:
    if not name:
        return set()
    return set(_WORD_RE.findall(name.lower()))


def _name_similar(a: Optional[str], b: Optional[str]) -> bool:
    """Token-set Jaccard ≥ 0.5, or either side missing → treat as similar.

    A near-coincident pair where one provider didn't fill a name is much
    more likely to be the same place than two distinct facilities, so the
    empty-name case is a positive match. Bare equality after normalisation
    short-circuits to True for the common 'St. Mary Hospital' vs.
    'St Mary Hospital' case.
    """
    if not a or not b:
        return True
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return True
    inter = len(ta & tb)
    union = len(ta | tb)
    return (inter / union) >= 0.5 if union else True


def _project_metric(gdf: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
    """Project a non-empty GDF to its local UTM zone for metric distance work."""
    if gdf is None or len(gdf) == 0:
        return None
    try:
        utm_crs = gdf.estimate_utm_crs()
    except Exception:
        # Fall back to Web Mercator; close-to-equator zones still resolve
        # to a few-metre error which is well inside _POI_DEDUP_RADIUS_M.
        utm_crs = "EPSG:3857"
    return gdf.to_crs(utm_crs)


def _dedup_and_union(
    overture_gdf: gpd.GeoDataFrame,
    overpass_gdf: gpd.GeoDataFrame,
    *,
    radius_m: float,
) -> gpd.GeoDataFrame:
    """Merge the two tiers, dropping spatial+name duplicates.

    An Overpass point within ``radius_m`` of an Overture point AND with a
    similar name is treated as the same facility — Overture's row wins
    (its taxonomy is more uniform), but ``data_source`` is upgraded to
    ``"overture+osm"`` so callers can see the union confirmation.
    """
    empty = gpd.GeoDataFrame(
        columns=["name", "amenity", "data_source", "geometry"], crs="EPSG:4326"
    )

    has_ovr = overture_gdf is not None and len(overture_gdf) > 0
    has_osm = overpass_gdf is not None and len(overpass_gdf) > 0

    if not has_ovr and not has_osm:
        return empty

    if has_ovr and not has_osm:
        out = overture_gdf.copy()
        out["data_source"] = "overture_pois"
        return out.reset_index(drop=True)

    if has_osm and not has_ovr:
        out = overpass_gdf.copy()
        out["data_source"] = "osm_overpass"
        return out.reset_index(drop=True)

    # Both have rows — run KDTree-based spatial join in projected metres.
    ovr_proj = _project_metric(overture_gdf)
    osm_proj = _project_metric(overpass_gdf.to_crs(overture_gdf.crs))

    ovr_out = overture_gdf.copy().reset_index(drop=True)
    ovr_out["data_source"] = "overture_pois"

    try:
        from scipy.spatial import cKDTree  # type: ignore
        import numpy as np

        ovr_xy = np.array(
            [(g.x, g.y) for g in ovr_proj.geometry], dtype=float
        )
        osm_xy = np.array(
            [(g.x, g.y) for g in osm_proj.geometry], dtype=float
        )
        tree = cKDTree(ovr_xy)
        dists, idxs = tree.query(osm_xy, k=1, distance_upper_bound=float(radius_m))

        keep_mask = []
        for osm_row_i, (dist, ovr_i) in enumerate(zip(dists, idxs)):
            if dist == float("inf") or ovr_i >= len(ovr_out):
                keep_mask.append(True)
                continue
            osm_name = overpass_gdf.iloc[osm_row_i].get("name")
            ovr_name = ovr_out.iloc[ovr_i].get("name")
            if _name_similar(ovr_name, osm_name):
                # Mark the matching Overture row as union-confirmed.
                ovr_out.at[ovr_i, "data_source"] = "overture+osm"
                keep_mask.append(False)
            else:
                keep_mask.append(True)
    except Exception as exc:
        # Without scipy/numpy or on an unexpected layout, fall back to a
        # naïve "keep everything" union — better to have duplicates than
        # to drop legitimate facilities silently.
        logger.warning(
            "POI dedup KDTree path failed (%s); concatenating without dedup", exc
        )
        keep_mask = [True] * len(overpass_gdf)

    osm_keep = overpass_gdf.iloc[keep_mask].copy().reset_index(drop=True)
    osm_keep["data_source"] = "osm_overpass"

    merged = gpd.GeoDataFrame(
        # Use pandas concat under the hood; geopandas preserves CRS as
        # long as both frames are EPSG:4326 (which they are at this point).
        __import__("pandas").concat(
            [ovr_out, osm_keep], ignore_index=True, sort=False
        ),
        geometry="geometry",
        crs="EPSG:4326",
    )
    return merged


def fetch_pois(
    boundary_gdf: gpd.GeoDataFrame,
    category: str,
) -> gpd.GeoDataFrame:
    """Fetch POIs for ``category`` from Overture ∪ OSM Overpass.

    Both tiers run in parallel; results are spatially deduplicated within
    ``_POI_DEDUP_RADIUS_M`` metres when names also match.  Raises
    ``DataFetchError`` only when **both** tiers come back empty (or fail).
    """
    if category not in OVERTURE_CATEGORIES and category not in OSM_AMENITY_TAGS:
        raise DataFetchError(
            f"Unknown POI category '{category}'. "
            f"Supported: {sorted(set(OVERTURE_CATEGORIES) | set(OSM_AMENITY_TAGS))}"
        )

    bounds = boundary_gdf.to_crs("EPSG:4326").total_bounds
    bbox = (bounds[0], bounds[1], bounds[2], bounds[3])

    empty = gpd.GeoDataFrame(
        columns=["name", "amenity", "geometry"], crs="EPSG:4326"
    )

    tier_results: dict = {"overture": empty, "overpass": empty}
    tier_errors: dict = {}

    def _run_overture():
        return fetch_pois_via_overture(bbox, category)

    def _run_overpass():
        return fetch_pois_via_overpass(bbox, category)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_run_overture): "overture",
            pool.submit(_run_overpass): "overpass",
        }
        for fut in concurrent.futures.as_completed(futures):
            label = futures[fut]
            try:
                tier_results[label] = fut.result()
            except Exception as exc:
                tier_errors[label] = str(exc)
                logger.warning(
                    "POI tier '%s' failed: %s — continuing with the other tier",
                    label, exc,
                )

    ovr_gdf = tier_results["overture"]
    osm_gdf = tier_results["overpass"]
    ovr_n = len(ovr_gdf) if ovr_gdf is not None else 0
    osm_n = len(osm_gdf) if osm_gdf is not None else 0

    if ovr_n == 0 and osm_n == 0:
        err_detail = "; ".join(f"{k}: {v}" for k, v in tier_errors.items())
        raise DataFetchError(
            f"No '{category}' POIs found for this area. "
            f"Overture returned 0 features and Overpass returned 0 features. "
            + (f"Tier errors: {err_detail}" if err_detail else
               "The category may not be represented in either provider for this region.")
        )

    merged = _dedup_and_union(ovr_gdf, osm_gdf, radius_m=_POI_DEDUP_RADIUS_M)
    merged = _clip_to_boundary(merged, boundary_gdf)
    merged.attrs["tier_counts"] = {
        "overture": int(ovr_n),
        "overpass": int(osm_n),
        "after_union": int(len(merged)),
    }
    if tier_errors:
        merged.attrs["tier_errors"] = tier_errors

    merged["candidates_are_synthetic"] = False
    logger.info(
        "POI fetch (%s): Overture=%d, Overpass=%d, after union+clip=%d",
        category, ovr_n, osm_n, len(merged),
    )
    return merged
