"""Geometry + scale validation helpers.

Applied by every boundary tier before returning, so downstream code can assume
the polygon is non-empty, valid, in EPSG:4326, with sane bbox.
"""

from __future__ import annotations

import logging
from typing import Tuple

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

from utils.scale_classifier import _SCALE_AREA_THRESHOLDS  # type: ignore[attr-defined]

from .errors import GeocodingError

logger = logging.getLogger(__name__)


def _repair(geom):
    """Return a valid Polygon/MultiPolygon, repairing if needed."""
    if geom.is_valid:
        return geom
    repaired = make_valid(geom)
    # make_valid can return GeometryCollection; keep only polygonal parts.
    if repaired.geom_type == "GeometryCollection":
        polys = [g for g in repaired.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return geom.buffer(0)
        if len(polys) == 1:
            return polys[0]
        # Merge polygons → MultiPolygon
        merged = []
        for p in polys:
            if p.geom_type == "MultiPolygon":
                merged.extend(p.geoms)
            else:
                merged.append(p)
        return MultiPolygon(merged)
    if repaired.geom_type in ("Polygon", "MultiPolygon"):
        return repaired
    # Last resort: zero-width buffer normalizes self-intersections
    return geom.buffer(0)


def validate_polygon(gdf: gpd.GeoDataFrame, *, source_label: str = "") -> gpd.GeoDataFrame:
    """Normalise + validate a boundary GeoDataFrame in place.

    - Non-empty, has geometry column.
    - CRS == EPSG:4326 (reprojects if another CRS is set).
    - Single-row: keep the first polygon/multipolygon.
    - geometry.is_valid: repaired via make_valid()/buffer(0) if not.
    - geom_type in {Polygon, MultiPolygon}.
    - Bounds sanity: finite, minx<maxx, miny<maxy, lat/lon in range.

    Raises:
        GeocodingError with a specific reason if the polygon is unusable.
    """
    label = f" [{source_label}]" if source_label else ""

    if gdf is None or len(gdf) == 0:
        raise GeocodingError(f"Validation{label}: empty GeoDataFrame")

    try:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception as exc:
        raise GeocodingError(f"Validation{label}: CRS handling failed: {exc}") from exc

    geom = gdf.geometry.iloc[0]
    if geom is None or geom.is_empty:
        raise GeocodingError(f"Validation{label}: geometry is empty")

    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise GeocodingError(
            f"Validation{label}: expected Polygon/MultiPolygon, "
            f"got {geom.geom_type}"
        )

    if not geom.is_valid:
        repaired = _repair(geom)
        if repaired is None or repaired.is_empty or repaired.geom_type not in ("Polygon", "MultiPolygon"):
            raise GeocodingError(
                f"Validation{label}: geometry invalid and could not be repaired"
            )
        logger.info(f"Boundary geometry{label} was invalid; repaired via make_valid().")
        gdf = gdf.copy()
        gdf.geometry = [repaired] + list(gdf.geometry.iloc[1:])
        geom = repaired

    minx, miny, maxx, maxy = geom.bounds
    if not (minx < maxx and miny < maxy):
        raise GeocodingError(
            f"Validation{label}: degenerate bbox ({minx},{miny},{maxx},{maxy})"
        )
    if not (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and
            -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0):
        raise GeocodingError(
            f"Validation{label}: bbox outside WGS-84 range "
            f"({minx},{miny},{maxx},{maxy})"
        )

    return gdf


def validate_scale_match(
    gdf: gpd.GeoDataFrame,
    scale: str,
    *,
    hard_reject_factor: float = 100.0,
) -> Tuple[bool, str]:
    """Return (ok, reason). ``ok=False`` with a reason when the polygon's area
    is *severely* off for the declared scale (off by more than
    ``hard_reject_factor`` from the expected range). This is the check that
    prevents a city query from silently accepting a country polygon.

    Uses degree-squared area from total_bounds — rough but dependency-free and
    consistent with utils/scale_classifier.validate_boundary_scale().
    """
    try:
        bounds = gdf.to_crs("EPSG:4326").total_bounds
        area_deg2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    except Exception:
        return True, ""  # can't measure → don't block

    lo, hi = _SCALE_AREA_THRESHOLDS.get(scale, (0.0, float("inf")))
    if lo <= area_deg2 <= hi:
        return True, ""

    # Severe mismatch check — only reject when area is off by > hard_reject_factor
    if hi != float("inf") and area_deg2 > hi * hard_reject_factor:
        return (
            False,
            f"fetched area {area_deg2:.4f} sq-deg >> scale '{scale}' upper bound "
            f"{hi:.4f} (factor > {hard_reject_factor:.0f}× too large)",
        )
    if lo > 0 and area_deg2 < lo / hard_reject_factor:
        return (
            False,
            f"fetched area {area_deg2:.6f} sq-deg << scale '{scale}' lower bound "
            f"{lo:.4f} (factor > {hard_reject_factor:.0f}× too small)",
        )

    # Moderate mismatch: soft-pass so the caller can warn the user but keep
    # the polygon (many real boundaries sit at the border between scales).
    return True, ""
