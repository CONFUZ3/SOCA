"""Overpass POI tier — companion to the Overture path in pois.py.

Used by ``fetch_pois`` as the second arm of an Overture ∪ Overpass union so
regions where Overture's place coverage is sparse (much of Africa, parts of
South Asia / Latin America) still resolve to non-empty facility data.

Returns an empty GeoDataFrame when Overpass succeeds but the bbox has no
matching features; raises ``DataFetchError`` only on transport / parse
failures.  ``utils.fetchers.http.make_request`` already handles 429 / 5xx /
timeout retries with exponential backoff, so the caller can treat
``DataFetchError`` here as terminal.
"""

from __future__ import annotations

import logging

import geopandas as gpd
from shapely.geometry import Point

from .constants import (
    OSM_AMENITY_TAGS,
    OVERPASS_URL,
    _OVERPASS_QUERY_TIMEOUT_SEC,
)
from .errors import DataFetchError
from .http import make_request

logger = logging.getLogger(__name__)


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        columns=["name", "amenity", "geometry"], crs="EPSG:4326"
    )


def _build_overpass_query(
    tags: list[tuple[str, str]],
    south: float, west: float, north: float, east: float,
) -> str:
    """Construct an Overpass QL body querying node/way/relation across all tag pairs.

    ``out center tags`` makes Overpass return a centroid for ways/relations
    so polygon features (hospital campuses) resolve to a single coordinate
    without a follow-up roundtrip.
    """
    bbox = f"{south},{west},{north},{east}"
    parts: list[str] = []
    for key, value in tags:
        # Overpass uses unquoted bareword keys/values; both are ASCII-safe
        # in our mapping so simple double-quoting is fine.
        sel = f'["{key}"="{value}"]'
        parts.append(f"  node{sel}({bbox});")
        parts.append(f"  way{sel}({bbox});")
        parts.append(f"  relation{sel}({bbox});")
    body = "\n".join(parts)
    return (
        f"[out:json][timeout:{_OVERPASS_QUERY_TIMEOUT_SEC}];\n"
        f"(\n{body}\n);\n"
        "out center tags;"
    )


def _element_point(element: dict):
    """Extract (lon, lat) from a node element, or the ``center`` of a way/relation."""
    if element.get("type") == "node":
        lon, lat = element.get("lon"), element.get("lat")
    else:
        center = element.get("center") or {}
        lon, lat = center.get("lon"), center.get("lat")
    if lon is None or lat is None:
        return None
    try:
        return Point(float(lon), float(lat))
    except (TypeError, ValueError):
        return None


def _matched_amenity(tags: dict, mapping: list[tuple[str, str]], default: str) -> str:
    """Return the OSM tag value that satisfied the query, falling back to the category."""
    for key, value in mapping:
        if tags.get(key) == value:
            return value
    return default


def _elements_to_gdf(elements: list[dict], category: str) -> gpd.GeoDataFrame:
    if not elements:
        return _empty_gdf()
    mapping = OSM_AMENITY_TAGS.get(category, [])
    rows: list[dict] = []
    geoms: list = []
    for el in elements:
        pt = _element_point(el)
        if pt is None:
            continue
        tags = el.get("tags") or {}
        name = tags.get("name") or tags.get("name:en") or ""
        amenity = _matched_amenity(tags, mapping, default=category)
        rows.append({"name": str(name), "amenity": str(amenity)})
        geoms.append(pt)
    if not rows:
        return _empty_gdf()
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


def fetch_pois_via_overpass(
    bbox: tuple,
    category: str,
) -> gpd.GeoDataFrame:
    """Fetch facility POIs from the OSM Overpass API for the given category.

    ``bbox`` is (west, south, east, north) in EPSG:4326, matching the
    convention used elsewhere in ``utils.fetchers``.

    Returns an empty GeoDataFrame when Overpass returns no matching elements.
    Raises ``DataFetchError`` when the request fails after the retry budget
    in ``http.make_request`` is exhausted, or when the response cannot be
    parsed.
    """
    tags = OSM_AMENITY_TAGS.get(category)
    if not tags:
        return _empty_gdf()

    west, south, east, north = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    body = _build_overpass_query(tags, south, west, north, east)

    try:
        resp = make_request(
            OVERPASS_URL,
            params={"data": body},
            method="POST",
            timeout=_OVERPASS_QUERY_TIMEOUT_SEC,
        )
    except DataFetchError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise DataFetchError(f"Overpass request failed: {exc}") from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise DataFetchError(f"Overpass returned non-JSON body: {exc}") from exc

    elements = payload.get("elements") or []
    gdf = _elements_to_gdf(elements, category)
    logger.info(
        "Overpass POIs (%s): %d elements → %d points after geometry filter",
        category, len(elements), len(gdf),
    )
    return gdf
