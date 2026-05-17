"""Custom Overpass plugin — fetch any OSM features by arbitrary tag dict."""

from __future__ import annotations

import logging

import geopandas as gpd
from shapely.geometry import Point

from .constants import OVERPASS_URL, _OVERPASS_QUERY_TIMEOUT_SEC
from .errors import DataFetchError
from .http import make_request
from .source_registry import DataSourcePlugin

logger = logging.getLogger(__name__)


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")


def _build_query(osm_tags: dict, south: float, west: float, north: float, east: float) -> str:
    bbox = f"{south},{west},{north},{east}"
    parts: list[str] = []
    for key, value in osm_tags.items():
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


class OverpassCustomPlugin(DataSourcePlugin):
    name = "overpass_custom"
    description = (
        "Fetch any OpenStreetMap feature type by OSM tags. "
        "Pass osm_tags= as a dict e.g. {'amenity': 'school'} or "
        "{'building': 'hospital'}."
    )
    supported_categories = ["facilities", "infrastructure", "other"]

    def validate_params(self, **kwargs) -> tuple[bool, str]:
        osm_tags = kwargs.get("osm_tags") or {}
        if not isinstance(osm_tags, dict) or not osm_tags:
            return False, "osm_tags must be a non-empty dict (e.g. {'amenity':'school'})"
        return True, ""

    def fetch(self, boundary_gdf: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        osm_tags = kwargs.get("osm_tags") or {}
        bounds = boundary_gdf.to_crs("EPSG:4326").total_bounds  # minx,miny,maxx,maxy
        west, south, east, north = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
        body = _build_query(osm_tags, south, west, north, east)

        try:
            resp = make_request(
                OVERPASS_URL,
                params={"data": body},
                method="POST",
                timeout=_OVERPASS_QUERY_TIMEOUT_SEC,
            )
        except Exception as exc:
            raise DataFetchError(f"Overpass request failed: {exc}") from exc

        try:
            payload = resp.json()
        except ValueError as exc:
            raise DataFetchError(f"Overpass returned non-JSON body: {exc}") from exc

        elements = payload.get("elements") or []
        if not elements:
            return _empty_gdf()

        default_amenity = next(iter(osm_tags.values()))
        rows: list[dict] = []
        geoms: list = []
        for el in elements:
            pt = _element_point(el)
            if pt is None:
                continue
            tags = el.get("tags") or {}
            name = tags.get("name") or tags.get("name:en") or ""
            amenity = next(
                (tags[k] for k in osm_tags if tags.get(k) == osm_tags[k]),
                default_amenity,
            )
            rows.append({"name": str(name), "amenity": str(amenity)})
            geoms.append(pt)

        if not rows:
            return _empty_gdf()
        gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
        logger.info(
            "Overpass custom (%s): %d elements → %d points",
            osm_tags, len(elements), len(gdf),
        )
        return gdf
