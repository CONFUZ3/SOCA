"""Boundary-polygon fetchers.

Tier order when no ``hint`` is supplied:
    Overture division/area → Nominatim /search

If an OSM id ``hint`` is supplied, a direct Nominatim /lookup is tried first
(~0.5 s), then the chain above if that fails.

Every tier's result is passed through ``validate_polygon`` (for validity/CRS
sanity) and ``validate_scale_match`` (to reject a country polygon returned
for a city query). A tier that returns a severely-wrong-scale polygon is
rejected, and the next tier runs.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Optional

import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union

from .constants import (
    PHOTON_URL,
    _OVERTURE_READ_TIMEOUT_SEC,
    _OVERTURE_SUBTYPE_ADMIN_LEVEL,
)
from .errors import DataFetchError, GeocodingError
from .http import make_request, nominatim_get
from .validation import validate_polygon, validate_scale_match

try:
    import overturemaps  # type: ignore
    import pyarrow  # type: ignore  # noqa: F401
    _OVERTURE_AVAILABLE = True
except ImportError:
    _OVERTURE_AVAILABLE = False

logger = logging.getLogger(__name__)


def _validate_and_accept(
    gdf: gpd.GeoDataFrame,
    *,
    scale: str,
    source_label: str,
) -> gpd.GeoDataFrame:
    """Run the standard validation chain; raise if the tier's result is unusable."""
    gdf = validate_polygon(gdf, source_label=source_label)
    ok, reason = validate_scale_match(gdf, scale)
    if not ok:
        raise GeocodingError(
            f"{source_label}: {reason} — rejecting and falling through."
        )
    return gdf


def fetch_boundary_via_nominatim_lookup(
    *,
    osm_type: str,
    osm_id: int,
    location: str,
) -> gpd.GeoDataFrame:
    """Direct OSM id → polygon via Nominatim /lookup (<1 s)."""
    type_map = {"R": "R", "W": "W", "N": "N",
                "relation": "R", "way": "W", "node": "N"}
    t_prefix = type_map.get(osm_type)
    if t_prefix is None:
        raise GeocodingError(
            f"Invalid osm_type '{osm_type}' (expected R/W/N)."
        )

    params = {
        "osm_ids": f"{t_prefix}{int(osm_id)}",
        "format": "geojson",
        "polygon_geojson": 1,
        "addressdetails": 1,
        "extratags": 1,
    }
    try:
        response = nominatim_get("/lookup", params=params, timeout=20)
    except DataFetchError as exc:
        raise GeocodingError(
            f"Network error in Nominatim lookup for {t_prefix}{osm_id}: {exc}"
        ) from exc

    try:
        data = response.json()
    except Exception as exc:
        raise GeocodingError(f"Invalid JSON from Nominatim lookup: {exc}") from exc

    features = data.get("features", [])
    if not features:
        raise GeocodingError(f"Nominatim /lookup returned no feature for {t_prefix}{osm_id}.")

    feature = features[0]
    try:
        geom = shape(feature["geometry"])
    except Exception as exc:
        raise GeocodingError(f"Could not parse geometry from Nominatim /lookup: {exc}") from exc

    if geom.is_empty:
        raise GeocodingError(f"Nominatim /lookup returned empty geometry for {t_prefix}{osm_id}.")
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise GeocodingError(
            f"Nominatim /lookup returned non-polygon ({geom.geom_type}) for {t_prefix}{osm_id}."
        )

    props = feature.get("properties", {}) or {}
    extratags = props.get("extratags") or {}
    address = props.get("address") or {}

    out_props = {
        "name": props.get("display_name") or props.get("name") or location,
        "location_query": location,
        "source": "nominatim_lookup",
        "osm_id": osm_id,
        "osm_type": t_prefix,
        "country": address.get("country", ""),
        "country_code": (address.get("country_code") or "").upper(),
    }
    if isinstance(extratags, dict) and extratags.get("population"):
        out_props["population"] = str(extratags["population"])

    return gpd.GeoDataFrame([out_props], geometry=[geom], crs="EPSG:4326")


def fetch_boundary_via_nominatim(location: str) -> gpd.GeoDataFrame:
    """Nominatim /search with polygon_geojson=1. Fallback tier (~1–2 s)."""
    from shapely.geometry import box

    params = {
        "q": location,
        "format": "geojson",
        "polygon_geojson": 1,
        "limit": 3,
        "addressdetails": 1,
        "extratags": 1,
    }
    try:
        response = nominatim_get("/search", params=params, timeout=30)
    except DataFetchError as exc:
        raise GeocodingError(
            f"Network error fetching boundary for '{location}': {exc}"
        ) from exc

    try:
        data = response.json()
    except Exception as exc:
        raise GeocodingError(
            f"Invalid JSON from Nominatim for '{location}': {exc}"
        ) from exc

    features = data.get("features", [])
    if not features:
        raise GeocodingError(
            f"Nominatim returned no results for '{location}'. "
            "Try a more specific place name."
        )

    polygon_features = [
        f for f in features
        if f.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon")
    ]
    if not polygon_features:
        first = features[0]
        bbox = first.get("bbox")
        if bbox and len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
            geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
            props = first.get("properties", {}) or {}
            props["source"] = "nominatim_bbox_fallback"
            return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")
        raise GeocodingError(
            f"Nominatim result for '{location}' has no polygon geometry "
            f"and no valid bbox."
        )

    feature = polygon_features[0]
    try:
        geom = shape(feature["geometry"])
    except Exception as exc:
        raise GeocodingError(
            f"Could not parse geometry from Nominatim result: {exc}"
        ) from exc

    raw_props = feature.get("properties", {}) or {}
    address = raw_props.get("address") or {}
    extratags = raw_props.get("extratags") or {}

    props: dict = {
        "name": raw_props.get("display_name") or raw_props.get("name") or location,
        "location_query": location,
        "source": "nominatim",
        "osm_id": raw_props.get("osm_id"),
        "osm_type": raw_props.get("osm_type", ""),
        "country": address.get("country", ""),
        "country_code": (address.get("country_code") or "").upper(),
    }
    if isinstance(extratags, dict) and extratags.get("population"):
        props["population"] = str(extratags["population"])

    return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")


def _geocode_to_lonlat(location: str) -> tuple[float, float]:
    """Return (lon, lat) for a place name. Tries Photon first, Nominatim fallback."""
    # Photon — faster, prefix-search friendly.
    try:
        resp = make_request(
            PHOTON_URL + "/api",
            params={"q": location, "limit": 1, "lang": "en"},
            timeout=15,
        )
        data = resp.json()
        features = data.get("features", [])
        if features:
            coords = features[0].get("geometry", {}).get("coordinates", [])
            if coords and len(coords) >= 2:
                return float(coords[0]), float(coords[1])
    except Exception as exc:
        logger.debug(f"Photon geocode failed for Overture bbox ({exc}); trying Nominatim.")

    # Nominatim fallback.
    try:
        resp = nominatim_get(
            "/search",
            params={"q": location, "format": "json", "limit": 1},
            timeout=15,
        )
        data = resp.json()
        if data:
            return float(data[0]["lon"]), float(data[0]["lat"])
    except Exception as exc:
        raise DataFetchError(
            f"Geocode for Overture bbox failed (both Photon and Nominatim): {exc}"
        ) from exc

    raise DataFetchError(f"No geocode result for '{location}'")


_ADMIN_SUBTYPES = ["locality", "county", "region", "country", "localadmin", "neighborhood"]

_SCALE_PREFERRED_SUBTYPES: dict[str, list[str]] = {
    "country": ["country"],
    "region": ["region", "county", "localadmin"],
    "city": ["locality", "localadmin", "county", "region"],
    "neighborhood": ["neighborhood", "locality", "localadmin"],
}


def _rank_divisions(
    df,
    *,
    admin_level: Optional[int],
    scale: str,
):
    """Return `df` sorted best-first so callers can try candidates in order."""
    df = df.copy()
    if admin_level is not None:
        df["_al_dist"] = df["subtype"].map(
            lambda s: abs(_OVERTURE_SUBTYPE_ADMIN_LEVEL.get(s, 8) - admin_level)
        )
        return df.sort_values("_al_dist")
    preferred = _SCALE_PREFERRED_SUBTYPES.get(scale, ["locality", "localadmin", "county"])
    df["_subtype_rank"] = df["subtype"].apply(
        lambda s: preferred.index(s) if s in preferred else len(preferred)
    )
    return df.sort_values("_subtype_rank")


def _fetch_via_duckdb(
    location: str,
    admin_level: Optional[int],
    scale: str,
) -> gpd.GeoDataFrame:
    """Primary Overture path: DuckDB SQL against the us-west-2 parquet."""
    import shapely.wkb as wkb
    from utils.scale_classifier import get_bbox_buffer
    from . import overture_duckdb as od

    lon, lat = _geocode_to_lonlat(location)
    buf = get_bbox_buffer(scale)
    bbox = (
        max(-180.0, lon - buf), max(-90.0, lat - buf),
        min(180.0, lon + buf),  min(90.0, lat + buf),
    )
    primary_query = location.split(",")[0].strip()

    div_df = od.query_divisions(
        bbox=bbox,
        subtypes=_ADMIN_SUBTYPES,
        name_query=primary_query,
    )
    if div_df is None or div_df.empty:
        raise DataFetchError(
            f"Overture/DuckDB division: no entity matched '{primary_query}' near '{location}'"
        )

    # Prefer exact-name matches over substring matches (Overture SQL LIKE
    # catches e.g. "Liman" when the user asked for "Lima"). Fall back to
    # the full match set only if no exact hit.
    pq_lc = primary_query.strip().lower()
    exact = div_df[div_df["name"].astype(str).str.lower() == pq_lc]
    if not exact.empty:
        div_df = exact

    ranked = _rank_divisions(div_df, admin_level=admin_level, scale=scale)

    # Not every division has a polygon (e.g. some localities are point-only).
    # Try candidates in rank order until we find one with a division_area row.
    best_div = None
    area_df = None
    last_reason: str = "no candidates"
    for _, row in ranked.head(5).iterrows():
        candidate_id = str(row.get("id", "") or "")
        if not candidate_id:
            continue
        try:
            # Pass bbox: enables parquet row-group pruning on the `bbox` column.
            result = od.query_division_area_by_id(candidate_id, bbox=bbox)
        except DataFetchError as exc:
            last_reason = f"area query failed for {candidate_id}: {exc}"
            continue
        if result is not None and not result.empty:
            best_div = row
            area_df = result
            break
        last_reason = f"no polygon for division_id={candidate_id} ({row.get('subtype','?')})"

    if area_df is None or best_div is None:
        raise DataFetchError(f"Overture/DuckDB division_area: {last_reason}")

    division_id = str(best_div["id"])
    try:
        geom = wkb.loads(bytes(area_df.iloc[0]["geometry"]))
    except Exception as exc:
        raise DataFetchError(
            f"Overture/DuckDB division_area: could not decode geometry: {exc}"
        ) from exc

    population_raw = best_div.get("population", None)
    best_subtype = str(best_div.get("subtype", ""))
    best_name = str(best_div.get("name", "") or primary_query)

    props = {
        "name": best_name,
        "location_query": location,
        "source": "overture_division",
        "subtype": best_subtype,
        "admin_level": str(_OVERTURE_SUBTYPE_ADMIN_LEVEL.get(best_subtype, "")),
        "population": str(int(population_raw)) if population_raw else "",
        "division_id": division_id,
    }
    return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")


def _fetch_via_pyclient(
    location: str,
    admin_level: Optional[int],
    scale: str,
) -> gpd.GeoDataFrame:
    """Fallback path when DuckDB is unavailable — uses the `overturemaps` client."""
    if not _OVERTURE_AVAILABLE:
        raise DataFetchError("overturemaps package not available")

    import shapely.wkb as wkb
    from utils.scale_classifier import get_bbox_buffer

    lon, lat = _geocode_to_lonlat(location)
    buf = get_bbox_buffer(scale)
    bbox = (
        max(-180.0, lon - buf), max(-90.0, lat - buf),
        min(180.0, lon + buf),  min(90.0, lat + buf),
    )

    def _read(theme: str):
        reader = overturemaps.record_batch_reader(theme, bbox=bbox)
        if reader is None:
            raise DataFetchError(f"Overture {theme} reader returned None")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(reader.read_all)
            try:
                return fut.result(timeout=_OVERTURE_READ_TIMEOUT_SEC)
            except concurrent.futures.TimeoutError:
                raise DataFetchError(
                    f"Overture {theme} read_all timed out after {_OVERTURE_READ_TIMEOUT_SEC}s"
                )

    div_table = _read("division")
    if div_table.num_rows == 0:
        raise DataFetchError(f"Overture division: no results near '{location}'")

    div_df = div_table.to_pandas()
    if "subtype" in div_df.columns:
        div_df = div_df[div_df["subtype"].isin(set(_ADMIN_SUBTYPES))].copy()
    if div_df.empty:
        raise DataFetchError(f"Overture division: no admin subtypes near '{location}'")

    def _name(n):
        return (n.get("primary", "") if isinstance(n, dict) else "") or ""

    primary_query = location.split(",")[0].strip().lower()
    div_df["name"] = div_df["names"].apply(_name)
    name_match = div_df[div_df["name"].str.lower().str.contains(
        primary_query, na=False, regex=False
    )].copy()
    if name_match.empty:
        raise DataFetchError(
            f"Overture division: no entity matched '{primary_query}' near '{location}'"
        )

    best_div = _rank_divisions(name_match, admin_level=admin_level, scale=scale).iloc[0]
    division_id = str(best_div.get("id", "") or "")
    best_subtype = str(best_div.get("subtype", ""))

    area_table = _read("division_area")
    if area_table.num_rows == 0:
        raise DataFetchError(f"Overture division_area: no results near '{location}'")

    area_df = area_table.to_pandas()
    matched = area_df[area_df["division_id"] == division_id] if division_id else None
    if matched is None or matched.empty:
        area_df["_area_name"] = area_df["names"].apply(_name)
        matched = area_df[area_df["_area_name"].str.lower().str.contains(
            primary_query, na=False, regex=False
        )]
        if matched.empty:
            raise DataFetchError(
                f"Overture division_area: no polygon for '{location}' "
                f"(division_id={division_id})"
            )

    try:
        geom = wkb.loads(matched.iloc[0]["geometry"])
    except Exception as exc:
        raise DataFetchError(
            f"Overture division_area: could not decode geometry: {exc}"
        ) from exc

    population_raw = best_div.get("population", None)
    props = {
        "name": best_div.get("name", "") or location,
        "location_query": location,
        "source": "overture_division",
        "subtype": best_subtype,
        "admin_level": str(_OVERTURE_SUBTYPE_ADMIN_LEVEL.get(best_subtype, "")),
        "population": str(int(population_raw)) if population_raw else "",
        "division_id": division_id,
    }
    return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")


def fetch_boundary_via_overture(
    location: str,
    admin_level: Optional[int] = None,
    scale: str = "city",
) -> gpd.GeoDataFrame:
    """Overture Maps boundary fetch (division → division_area).

    Primary path is DuckDB SQL against ``s3://overturemaps-us-west-2`` per
    https://docs.overturemaps.org/guides/divisions/ — predicate pushdown
    makes this dramatically faster than the Python client, which has no
    server-side filtering beyond bbox. Falls back to the `overturemaps`
    client when DuckDB is unavailable.
    """
    from . import overture_duckdb as od

    if od.is_available():
        return _fetch_via_duckdb(location, admin_level, scale)
    logger.warning(
        "duckdb not installed — falling back to slow overturemaps client. "
        "Install with `pip install duckdb` for the fast path."
    )
    return _fetch_via_pyclient(location, admin_level, scale)


def fetch_boundaries(
    location: str,
    admin_level: Optional[int] = None,
    scale: str = "city",
    *,
    hint: Optional[object] = None,
) -> gpd.GeoDataFrame:
    """Run the boundary chain with validation between tiers.

    Tier 0 (hint): Nominatim /lookup by OSM id — fast (~1s) when the
                   geocoder provided an id hint.
    Tier 1:        Overture division + division_area.
    Tier 2:        Nominatim /search fallback.
    """
    from utils.activity_log import timed

    logger.info(
        f"fetch_boundaries: '{location}' (scale={scale}, "
        f"admin_level={admin_level}, hint={hint is not None})"
    )
    from utils.activity_log import log_event
    log_event(
        "boundary.fetch", "info",
        f'"{location}" (scale={scale}, admin_level={admin_level})',
    )

    last_exc: Optional[Exception] = None

    # Tier 0: direct OSM id via hint.
    hint_osm_id = getattr(hint, "osm_id", None) if hint is not None else None
    hint_osm_type = getattr(hint, "osm_type", None) if hint is not None else None
    if hint_osm_id and hint_osm_type in ("R", "W", "N"):
        try:
            with timed("boundary.fetch", source="Nominatim/id",
                       detail=f"{hint_osm_type}{hint_osm_id}") as t:
                gdf = fetch_boundary_via_nominatim_lookup(
                    osm_type=hint_osm_type, osm_id=int(hint_osm_id),
                    location=location,
                )
                gdf = validate_polygon(gdf, source_label="Nominatim/id")
                # Scale validation intentionally skipped here — user-disambiguated.
                t.detail = f"{hint_osm_type}{hint_osm_id} · {gdf.geometry.iloc[0].geom_type}"
            return gdf
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"Nominatim id lookup failed for {hint_osm_type}{hint_osm_id}: {exc}."
            )

    # Tier 1: Overture.
    if _OVERTURE_AVAILABLE:
        try:
            with timed("boundary.fetch", source="Overture", detail=location) as t:
                gdf = fetch_boundary_via_overture(location, admin_level=admin_level, scale=scale)
                gdf = _validate_and_accept(gdf, scale=scale, source_label="Overture")
                t.detail = f"{location} · {gdf.geometry.iloc[0].geom_type}"
            return gdf
        except Exception as exc:
            last_exc = exc
            logger.warning(f"Overture boundary failed for '{location}': {exc}")

    # Tier 2: Nominatim /search.
    try:
        with timed("boundary.fetch", source="Nominatim", detail=location) as t:
            gdf = fetch_boundary_via_nominatim(location)
            gdf = _validate_and_accept(gdf, scale=scale, source_label="Nominatim")
            t.detail = f"{location} · {gdf.geometry.iloc[0].geom_type}"
        return gdf
    except Exception as exc:
        last_exc = exc
        logger.warning(f"Nominatim boundary failed for '{location}': {exc}")

    raise GeocodingError(
        f"All geocoding backends failed for '{location}'. Last error: {last_exc}"
    )
