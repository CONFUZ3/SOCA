"""HDX generic-search plugin — query the CKAN API and download a dataset.

Searches https://data.humdata.org/api/3/action/package_search by keyword and
downloads the first usable CSV/GeoJSON resource. CSVs are point-ified by
locating common lat/lon column pairs. Anything else (zips, shapefiles,
Excel) is skipped because parsing them safely in-process is out of scope.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from shapely.ops import unary_union

from .errors import DataFetchError
from .http import make_request
from .source_registry import DataSourcePlugin

logger = logging.getLogger(__name__)

_HDX_API = "https://data.humdata.org/api/3/action/package_search"
_HDX_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB safety cap
_HDX_DOWNLOAD_TIMEOUT = 60

_LAT_CANDIDATES = ("lat", "latitude", "y", "lat_dd", "lat_y")
_LON_CANDIDATES = ("lon", "lng", "long", "longitude", "x", "lon_dd", "long_x")


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=["name", "source", "geometry"], crs="EPSG:4326")


def _pick_resource(resources: list[dict]) -> Optional[dict]:
    for r in resources:
        fmt = (r.get("format") or "").strip().lower()
        url = (r.get("url") or "").strip()
        if not url:
            continue
        if fmt in ("geojson", "json") or url.lower().endswith((".geojson", ".json")):
            return {"url": url, "kind": "geojson"}
        if fmt == "csv" or url.lower().endswith(".csv"):
            return {"url": url, "kind": "csv"}
    return None


def _find_latlon_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    cols_lower = {c.lower(): c for c in df.columns}
    lat_col = next((cols_lower[c] for c in _LAT_CANDIDATES if c in cols_lower), None)
    lon_col = next((cols_lower[c] for c in _LON_CANDIDATES if c in cols_lower), None)
    return lat_col, lon_col


def _stream_download(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=_HDX_DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()
    except Exception as exc:
        raise DataFetchError(f"HDX resource download failed: {exc}") from exc
    buf = bytearray()
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        buf.extend(chunk)
        if len(buf) > _HDX_MAX_DOWNLOAD_BYTES:
            raise DataFetchError(
                f"HDX resource exceeds {_HDX_MAX_DOWNLOAD_BYTES} byte cap; skipping."
            )
    return bytes(buf)


def _parse_geojson(blob: bytes) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(io.BytesIO(blob))
    except Exception as exc:
        raise DataFetchError(f"Could not parse GeoJSON: {exc}") from exc
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _parse_csv(blob: bytes) -> gpd.GeoDataFrame:
    try:
        df = pd.read_csv(io.BytesIO(blob))
    except Exception as exc:
        raise DataFetchError(f"Could not parse CSV: {exc}") from exc
    lat_col, lon_col = _find_latlon_columns(df)
    if not (lat_col and lon_col):
        raise DataFetchError(
            "CSV resource has no recognisable lat/lon columns "
            f"(tried {_LAT_CANDIDATES} / {_LON_CANDIDATES})."
        )
    df = df.dropna(subset=[lat_col, lon_col])
    geoms = [Point(float(x), float(y)) for x, y in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")


def _clip_to_boundary(gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        boundary_union = unary_union(boundary_gdf.to_crs("EPSG:4326").geometry.values)
    except Exception as exc:
        logger.warning("HDX clip union failed: %s; returning unclipped.", exc)
        return gdf
    try:
        return gdf[gdf.geometry.intersects(boundary_union)].reset_index(drop=True)
    except Exception as exc:
        logger.warning("HDX clip intersect failed: %s; returning unclipped.", exc)
        return gdf


class HDXGenericPlugin(DataSourcePlugin):
    name = "hdx_generic"
    description = (
        "Search HDX (Humanitarian Data Exchange) by keyword and download any "
        "dataset. Pass query= for the search term (e.g. 'flood risk Kenya'). "
        "Returns the first usable CSV/GeoJSON resource clipped to the AOI."
    )
    supported_categories = ["hazard", "health", "infrastructure", "population", "other"]

    def validate_params(self, **kwargs) -> tuple[bool, str]:
        query = (kwargs.get("query") or "").strip()
        if not query:
            return False, "query is required (e.g. 'flood risk Kenya')"
        return True, ""

    def fetch(self, boundary_gdf: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        query = (kwargs.get("query") or "").strip()
        resp = make_request(_HDX_API, params={"q": query, "rows": 5}, timeout=30)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise DataFetchError(f"HDX API returned non-JSON body: {exc}") from exc

        if not payload.get("success"):
            raise DataFetchError(f"HDX API search failed for query={query!r}")

        results = (payload.get("result") or {}).get("results") or []
        for pkg in results:
            picked = _pick_resource(pkg.get("resources") or [])
            if not picked:
                continue
            try:
                blob = _stream_download(picked["url"])
            except DataFetchError as exc:
                logger.warning("Skipping HDX resource: %s", exc)
                continue
            try:
                if picked["kind"] == "geojson":
                    gdf = _parse_geojson(blob)
                else:
                    gdf = _parse_csv(blob)
            except DataFetchError as exc:
                logger.warning("Skipping unparseable HDX resource: %s", exc)
                continue
            if gdf.empty:
                continue
            return _clip_to_boundary(gdf, boundary_gdf)

        raise DataFetchError(
            f"No usable CSV/GeoJSON resource found on HDX for query={query!r}."
        )
