"""
Automatic data fetching utilities for SOCA.

Retrieves geographic boundaries, POIs, and synthetic population grids
from public APIs (OpenStreetMap Nominatim, Overpass) so the Gemini agent
can work without manual data uploads.

Design principles:
- No new required dependencies: uses requests (transitive), geopandas, shapely
  which are already in the SOCA environment.
- Per-step failure isolation: callers iterate through steps and catch
  DataFetchError per step so partial fetches are still usable.
- 1-second rate-limit delay on Nominatim requests (ToS requirement).
- Exponential back-off retry (1s, 2s, 4s) for transient network errors.
"""

from __future__ import annotations

import logging
import re
import time
import random
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, shape, box, MultiPolygon
from shapely.ops import unary_union

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    import overturemaps
    import pyarrow as pa
    import pyarrow.compute as pc
    _OVERTURE_AVAILABLE = True
except ImportError:
    _OVERTURE_AVAILABLE = False

try:
    import pygadm  # optional — used only for final-tier admin boundary lookup
    _GADM_AVAILABLE = True
except ImportError:
    _GADM_AVAILABLE = False

from utils.activity_log import log_event, timed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DataFetchError(Exception):
    """Base exception for all data-fetching failures."""


class GeocodingError(DataFetchError):
    """Raised when Nominatim geocoding / boundary retrieval fails."""


class PopulationDataError(DataFetchError):
    """Raised when synthetic population-grid generation fails."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOMINATIM_URL = "https://nominatim.openstreetmap.org"
HDX_BASE_URL = "https://data.humdata.org/api/3/action"
PHOTON_URL = "https://photon.komoot.io"

# Shared User-Agent — Nominatim ToS requires an app-specific UA, not a library
# default. Mirrors utils/geocoder.py:_USER_AGENT so both endpoints identify as
# the same SOCA app. Also handed to osmnx via ox.settings.http_user_agent.
_USER_AGENT = (
    "SOCA/1.0 (Spatial Optimization Conversational Agent; "
    "academic research; contact: soca@example.com)"
)

# Overture Maps Place Category Mapping (Singular forms as per 2024/2025 taxonomy)
OVERTURE_CATEGORIES: dict[str, list[str]] = {
    "health": ["hospital", "medical_clinic", "doctor", "pharmacy", "medical_center", "health_center"],
    "education": ["school", "university", "college", "kindergarten", "preschool"],
    "food": ["supermarket", "grocery_store", "convenience_store", "market"],
    "finance": ["bank", "atm"],
    "fire_station": ["fire_station"],
    "police": ["police_station"],
    "library": ["library"],
    "transport": ["bus_stop", "train_station", "subway_station", "ferry_terminal", "airport", "transit_stop"],
    "water": ["water_point", "water_well", "water_treatment_plant", "drinking_water"],
    "emergency": ["emergency_shelter", "evacuation_center", "civil_defense"],
}

# Maps Overture division subtype to approximate OSM admin_level equivalent
_OVERTURE_SUBTYPE_ADMIN_LEVEL: dict[str, int] = {
    "country": 2,
    "region": 4,
    "county": 6,
    "localadmin": 7,
    "locality": 8,
    "neighborhood": 10,
}

# OSMnx tag filters — dict-of-lists form expected by ox.features_from_polygon.
_OSMNX_POI_TAGS: dict[str, dict] = {
    "health":       {"amenity": ["hospital", "clinic", "doctors", "pharmacy", "health_centre"]},
    "education":    {"amenity": ["school", "university", "college", "kindergarten"]},
    "food":         {"amenity": ["marketplace", "supermarket"],
                     "shop":    ["supermarket", "convenience", "grocery"]},
    "finance":      {"amenity": ["bank", "atm"]},
    "fire_station": {"amenity": "fire_station"},
    "police":       {"amenity": "police"},
    "library":      {"amenity": "library"},
    "transport":    {"public_transport": True,
                     "railway":          ["station", "halt"],
                     "amenity":          ["bus_station", "ferry_terminal"]},
    "water":        {"amenity":  ["water_point", "drinking_water"],
                     "man_made": ["water_well", "water_works"]},
    "emergency":    {"amenity":   "shelter",
                     "emergency": ["assembly_point", "evacuation_centre"]},
}

# Default total synthetic population spread over all grid points
_DEFAULT_TOTAL_POPULATION = 100_000

# Retry parameters
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1  # seconds — doubles each retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_bbox_only(gdf: gpd.GeoDataFrame) -> bool:
    """True when the returned boundary is just a rectangle (bbox fallback)."""
    if gdf is None or len(gdf) == 0:
        return True
    src = ""
    try:
        src = str(gdf.iloc[0].get("source", ""))
    except Exception:
        pass
    return "bbox_fallback" in src


# ---------------------------------------------------------------------------
# DataFetcher class
# ---------------------------------------------------------------------------

class DataFetcher:
    """
    High-level interface for automatic geographic data retrieval.

    Usage example::

        fetcher = DataFetcher()
        boundary = fetcher.fetch_boundaries("Lima, Peru")
        population = fetcher.fetch_population(boundary)
        hospitals  = fetcher.fetch_pois(boundary, "health")
    """

    def __init__(self) -> None:
        if not _REQUESTS_AVAILABLE:
            raise ImportError(
                "The 'requests' package is required for DataFetcher. "
                "Install it with: pip install requests"
            )
        self._last_nominatim_call: float = 0.0  # epoch seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_boundaries(
        self,
        location: str,
        admin_level: Optional[int] = None,
        scale: str = "city",
        *,
        hint: Optional[object] = None,
        prefer_polygon: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Fetch the administrative boundary polygon for *location*.

        Fallback chain (no Overpass direct):
          -1. OSMnx by OSM id     (if hint supplied an osm_id + osm_type)
           0. Overture Maps       (scale-aware, real population metadata)
           1. OSMnx by place name (ox.geocode_to_gdf — Nominatim+Overpass wrapped)
           2. Nominatim /search   (polygon_geojson=1, ToS-compliant 1 req/s)
           3. Photon bbox         (rectangular fallback)
           4. GADM                (last resort; country/state admin levels)

        Args:
            location:    Human-readable place name, e.g. ``"Lima, Peru"``.
            admin_level: Target OSM admin_level integer (2–10). When provided,
                         the Overture backend prefers the administrative entity
                         whose level is closest to this value, avoiding
                         mis-matches (e.g. returning a country polygon when a
                         city was requested).
            scale:       One of "country", "region", "city", "neighborhood".
                         Controls the Overture bbox buffer size so the query
                         covers the right geographic extent.

        Returns:
            GeoDataFrame with one row containing the boundary polygon.
            CRS is WGS-84 (EPSG:4326).

        Raises:
            GeocodingError: If all backends fail or return no polygon.
        """
        logger.info(
            f"DataFetcher: Fetching boundary for '{location}' "
            f"(scale={scale}, admin_level={admin_level}, hint={hint is not None})"
        )
        log_event(
            "boundary.fetch", "info",
            f'"{location}" (scale={scale}, admin_level={admin_level})',
        )

        # --- Tier -1: Direct OSM relation lookup via hint -----------------
        # When the caller has already disambiguated the place (e.g. from the
        # AOI picker's autocomplete), hand osmnx the prefixed OSM id directly.
        # osmnx caches the Overpass response, handles retries, and returns a
        # proper polygon — no hand-rolled mirror rotation needed.
        hint_osm_id = getattr(hint, "osm_id", None) if hint is not None else None
        hint_osm_type = getattr(hint, "osm_type", None) if hint is not None else None
        if hint_osm_id and hint_osm_type in ("R", "W", "N"):
            try:
                gdf = self._fetch_boundary_via_osmnx(
                    location, osm_type=hint_osm_type, osm_id=hint_osm_id
                )
                return gdf
            except Exception as exc:
                logger.warning(
                    f"OSMnx id lookup failed for {hint_osm_type}{hint_osm_id}: {exc}. "
                    "Falling back to full tier chain."
                )

        # --- Tier 0: Overture Maps division + division_area (primary) -------
        if _OVERTURE_AVAILABLE:
            try:
                with timed("boundary.fetch", source="Overture", detail=location) as t:
                    gdf = self._fetch_boundary_via_overture(
                        location, admin_level=admin_level, scale=scale
                    )
                    t.detail = f'{location} · {gdf.geometry.iloc[0].geom_type}'
                return gdf
            except Exception as exc:
                logger.warning(
                    f"Overture boundary fetch failed for '{location}': {exc}. "
                    "Falling back to Photon."
                )
                log_event(
                    "boundary.fetch", "fail", str(exc)[:140], source="Overture",
                )

        # --- Tier 1: OSMnx by place name ---------------------------------
        # osmnx.geocode_to_gdf wraps Nominatim + Overpass with proper caching,
        # retries, and the Overpass rate limit — no mirror rotation needed.
        try:
            gdf = self._fetch_boundary_via_osmnx(location)
            return gdf
        except Exception as exc:
            logger.warning(
                f"OSMnx boundary fetch failed for '{location}': {exc}. "
                "Falling back to Nominatim."
            )

        # --- Tier 2: Nominatim /search with polygon_geojson=1 -------------
        try:
            with timed("boundary.fetch", source="Nominatim", detail=location) as t:
                gdf = self._fetch_boundary_via_nominatim(location)
                t.detail = f'{location} · {gdf.geometry.iloc[0].geom_type}'
            return gdf
        except Exception as exc:
            logger.warning(
                f"Nominatim boundary fetch failed for '{location}': {exc}. "
                "Falling back to Photon bbox."
            )
            log_event("boundary.fetch", "fail", str(exc)[:140], source="Nominatim")

        # --- Tier 3: Photon bbox (rectangular fallback) -------------------
        try:
            with timed("boundary.fetch", source="Photon", detail=location) as t:
                gdf = self._fetch_boundary_via_photon(location)
                src = gdf.iloc[0].get("source", "photon") if len(gdf) else "photon"
                t.detail = f'{location} · {gdf.geometry.iloc[0].geom_type} · via {src}'
            return gdf
        except Exception as exc:
            logger.warning(
                f"Photon bbox fetch failed for '{location}': {exc}. Falling back to GADM."
            )
            log_event("boundary.fetch", "fail", str(exc)[:140], source="Photon")

        # --- Tier 4: GADM (last resort, country / state only) -------------
        if _GADM_AVAILABLE:
            try:
                with timed("boundary.fetch", source="GADM", detail=location) as t:
                    gdf = self._fetch_boundary_via_gadm(location, admin_level=admin_level)
                    t.detail = f'{location} · {gdf.geometry.iloc[0].geom_type}'
                return gdf
            except Exception as exc:
                log_event("boundary.fetch", "fail", str(exc)[:140], source="GADM")
                raise GeocodingError(
                    f"All geocoding backends (incl. GADM) failed for '{location}'. "
                    f"Last error: {exc}"
                ) from exc

        raise GeocodingError(
            f"All geocoding backends failed for '{location}'."
        )

    # ------------------------------------------------------------------
    # Boundary backend implementations
    # ------------------------------------------------------------------

    def _configure_osmnx_once(self):
        """Idempotent osmnx.settings configuration. Returns the osmnx module."""
        import osmnx as ox
        if getattr(self, "_osmnx_configured", False):
            return ox
        ox.settings.use_cache = True
        ox.settings.requests_timeout = 180
        ox.settings.overpass_rate_limit = True
        ox.settings.log_console = False
        # Per Nominatim/Overpass ToS: identify as this specific app.
        ox.settings.http_user_agent = _USER_AGENT
        self._osmnx_configured = True
        return ox

    def _fetch_boundary_via_osmnx(
        self,
        location: str,
        *,
        osm_type: Optional[str] = None,   # "R" | "W" | "N"
        osm_id: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """Fetch an admin-boundary polygon via osmnx.geocode_to_gdf.

        When *osm_type* + *osm_id* are supplied (e.g. from the AOI picker's
        geocoder autocomplete), we hand osmnx the prefixed id (``R358002``)
        and it skips disambiguation. Otherwise we pass the free-text location.
        """
        ox = self._configure_osmnx_once()

        source = "OSMnx/id" if (osm_type and osm_id) else "OSMnx"
        detail = f"{osm_type}{osm_id}" if (osm_type and osm_id) else location
        with timed("boundary.fetch", source=source, detail=detail) as t:
            if osm_type and osm_id:
                # osmnx 2.x requires type-prefixed queries: "R358002".
                query = f"{osm_type}{int(osm_id)}"
                gdf = ox.geocode_to_gdf([query], by_osmid=True)
            else:
                gdf = ox.geocode_to_gdf(location, which_result=1)

            if gdf is None or len(gdf) == 0:
                raise GeocodingError(f"OSMnx returned no polygon for '{location}'")

            gdf = gdf.to_crs("EPSG:4326")
            geom = gdf.geometry.iloc[0]
            if geom is None or geom.is_empty:
                raise GeocodingError(f"OSMnx returned empty geometry for '{location}'")

            row0 = gdf.iloc[0]
            props = {
                "name": str(row0.get("display_name", location)),
                "location_query": location,
                "source": "osmnx",
                "osm_id": osm_id if osm_id else row0.get("osm_id"),
                "osm_type": osm_type or row0.get("osm_type", ""),
            }
            t.detail = f"{detail} · {geom.geom_type}"
            return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")

    def _fetch_boundary_via_gadm(
        self,
        location: str,
        admin_level: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """Fetch admin boundary from GADM (final fallback, requires ``pygadm``).

        GADM is the gold-standard open admin-boundary dataset for levels 0–3
        (country / state / county). Only runs when all OSM-backed tiers have
        failed and ``pygadm`` is installed.
        """
        if not _GADM_AVAILABLE:
            raise DataFetchError("pygadm not installed; GADM tier unavailable")

        # pygadm exposes Items() to query by name across admin levels.
        # Heuristic: country-like queries (short, single-token) → level 0;
        # anything else → level 1. We don't have a perfect way to resolve
        # nested places without geocoding, so keep this tier deliberately
        # conservative — users asking for a specific city shouldn't reach it.
        level = 0 if admin_level is not None and admin_level <= 2 else 1
        try:
            gdf = pygadm.Items(name=location, admin=level)
        except Exception as exc:
            raise DataFetchError(f"pygadm lookup failed: {exc}") from exc

        if gdf is None or len(gdf) == 0:
            raise GeocodingError(f"GADM returned no results for '{location}'")

        # Normalise output schema to match other tiers.
        gdf = gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf
        name_col = next(
            (c for c in ("NAME_1", "NAME_0", "name") if c in gdf.columns),
            None,
        )
        name = str(gdf.iloc[0][name_col]) if name_col else location
        props = {
            "name": name,
            "location_query": location,
            "source": "gadm",
            "admin_level": str(level),
        }
        return gpd.GeoDataFrame(
            [props], geometry=[gdf.geometry.iloc[0]], crs="EPSG:4326"
        )

    def _fetch_boundary_via_photon(self, location: str) -> gpd.GeoDataFrame:
        """Fetch a rectangular-bbox boundary via Photon (photon.komoot.io).

        Used as a last-resort fallback once OSMnx and Nominatim have failed.
        Returns a plain bbox polygon (or small point buffer) — a rectangle
        often covers ocean near coasts, so callers should treat this as low
        quality and prefer upstream tiers when available.
        """
        params = {"q": location, "limit": 5, "lang": "en"}
        try:
            resp = self._make_request(PHOTON_URL + "/api", params=params, timeout=20)
        except DataFetchError as exc:
            raise GeocodingError(f"Photon request failed: {exc}") from exc

        try:
            data = resp.json()
        except Exception as exc:
            raise GeocodingError(f"Photon returned invalid JSON: {exc}") from exc

        features = data.get("features", [])
        if not features:
            raise GeocodingError(
                f"Photon returned no results for '{location}'."
            )

        best = None
        for feat in features:
            p = feat.get("properties", {})
            if p.get("extent"):
                best = feat
                break
        if best is None:
            best = features[0]

        props_raw = best.get("properties", {})
        extent = props_raw.get("extent")  # [minx, miny, maxx, maxy]
        if extent and len(extent) == 4:
            geom = box(extent[0], extent[1], extent[2], extent[3])
            logger.warning(
                f"DataFetcher: Using rectangular bbox polygon for '{location}' — "
                "points near coast may fall in water."
            )
        else:
            # Fall back to point + small buffer (0.05 degrees ~5 km)
            coords = best.get("geometry", {}).get("coordinates", [])
            if not coords or len(coords) < 2:
                raise GeocodingError(
                    f"Photon result for '{location}' has no usable geometry."
                )
            geom = Point(coords[0], coords[1]).buffer(0.05, resolution=16)

        props = {
            "name": props_raw.get("name", location),
            "location_query": location,
            "source": "photon_bbox_fallback",
            "country": props_raw.get("country", ""),
        }
        return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")

    def _fetch_boundary_via_nominatim(self, location: str) -> gpd.GeoDataFrame:
        """Original Nominatim boundary fetch (kept as last-resort fallback)."""
        self._nominatim_rate_limit()

        params = {
            "q": location,
            "format": "geojson",
            "polygon_geojson": 1,
            "limit": 1,
            "addressdetails": 0,
            "extratags": 1,
        }

        try:
            response = self._make_request(NOMINATIM_URL + "/search", params=params)
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
            if bbox:
                geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
                props = first.get("properties", {})
                props["source"] = "nominatim_bbox_fallback"
                return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")
            raise GeocodingError(
                f"Nominatim result for '{location}' has no polygon geometry."
            )

        feature = polygon_features[0]
        try:
            geom = shape(feature["geometry"])
        except Exception as exc:
            raise GeocodingError(
                f"Could not parse geometry from Nominatim result: {exc}"
            ) from exc

        props = feature.get("properties", {})
        props["location_query"] = location
        props["source"] = "nominatim"
        # Extract population from Nominatim extratags (available when extratags=1)
        extratags = props.get("extratags") or {}
        if isinstance(extratags, dict) and extratags.get("population"):
            props["population"] = extratags["population"]
        return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")

    def _fetch_boundary_via_overture(
        self,
        location: str,
        admin_level: Optional[int] = None,
        scale: str = "city",
    ) -> gpd.GeoDataFrame:
        """Fetch administrative boundary polygon from Overture Maps.

        Two-step process:
        1. Query the ``division`` theme to find the matching entity and read its
           real ``population`` field plus the ``id`` used to join to geometry.
        2. Query the ``division_area`` theme and match by ``division_id`` to get
           the actual polygon geometry.

        The bbox size is scale-aware (country=±15°, region=±5°, city=±2°,
        neighborhood=±0.5°) so the query covers the right extent regardless of
        geographic size. Name-matching uses both ``primary`` and ``common``
        name fields to handle non-English place names globally.
        """
        if not _OVERTURE_AVAILABLE:
            raise DataFetchError("overturemaps package not available")

        import shapely.wkb as wkb
        from utils.scale_classifier import get_bbox_buffer

        # ------------------------------------------------------------------ #
        # Step 1: Geocode location → (lon, lat) seed via Photon              #
        # ------------------------------------------------------------------ #
        params = {"q": location, "limit": 1, "lang": "en"}
        try:
            resp = self._make_request(PHOTON_URL + "/api", params=params, timeout=15)
            data = resp.json()
        except Exception as exc:
            raise DataFetchError(f"Photon geocode for Overture bbox failed: {exc}") from exc

        features = data.get("features", [])
        if not features:
            raise DataFetchError(f"Photon returned no results for '{location}'")

        coords = features[0].get("geometry", {}).get("coordinates", [])
        if not coords or len(coords) < 2:
            raise DataFetchError(f"Photon result has no coordinates for '{location}'")

        lon, lat = float(coords[0]), float(coords[1])
        buf = get_bbox_buffer(scale)
        bbox = (
            max(-180.0, lon - buf),
            max(-90.0,  lat - buf),
            min(180.0,  lon + buf),
            min(90.0,   lat + buf),
        )

        logger.info(
            f"DataFetcher: Querying Overture division for '{location}' "
            f"bbox={bbox} (scale={scale}, buf={buf}°)"
        )

        # ------------------------------------------------------------------ #
        # Step 2: Query Overture 'division' theme for metadata + population  #
        # ------------------------------------------------------------------ #
        ADMIN_SUBTYPES = {"locality", "county", "region", "country", "localadmin", "neighborhood"}

        try:
            div_reader = overturemaps.record_batch_reader("division", bbox=bbox)
            if div_reader is None:
                raise DataFetchError("Overture division reader returned None")
            div_table = div_reader.read_all()
        except Exception as exc:
            raise DataFetchError(f"Overture division query failed: {exc}") from exc

        if div_table.num_rows == 0:
            raise DataFetchError(f"Overture division: no results near '{location}'")

        div_df = div_table.to_pandas()

        if "subtype" in div_df.columns:
            div_df = div_df[div_df["subtype"].isin(ADMIN_SUBTYPES)].copy()

        if div_df.empty:
            raise DataFetchError(f"Overture division: no admin subtypes near '{location}'")

        def _extract_names(names_val) -> list[str]:
            """Return all name strings from Overture names struct (primary + common)."""
            if not isinstance(names_val, dict):
                return []
            result = []
            if names_val.get("primary"):
                result.append(str(names_val["primary"]))
            common = names_val.get("common") or {}
            if isinstance(common, dict):
                result.extend(str(v) for v in common.values() if v)
            return result

        primary_query = location.split(",")[0].strip().lower()

        div_df["_all_names"] = div_df["names"].apply(_extract_names)
        div_df["_name_match"] = div_df["_all_names"].apply(
            lambda names: any(primary_query in n.lower() for n in names)
        )
        name_match = div_df[div_df["_name_match"]].copy()

        if name_match.empty:
            raise DataFetchError(
                f"Overture division: no entity matched '{primary_query}' near '{location}'"
            )

        # Pick the entity whose subtype admin_level is closest to the hint
        _SCALE_PREFERRED_SUBTYPES: dict[str, list[str]] = {
            "country": ["country"],
            "region": ["region", "county", "localadmin"],
            "city": ["locality", "localadmin", "county", "region"],
            "neighborhood": ["neighborhood", "locality", "localadmin"],
        }
        if admin_level is not None:
            name_match["_al_dist"] = name_match["subtype"].map(
                lambda s: abs(_OVERTURE_SUBTYPE_ADMIN_LEVEL.get(s, 8) - admin_level)
            )
            best_div = name_match.sort_values("_al_dist").iloc[0]
        else:
            preferred = _SCALE_PREFERRED_SUBTYPES.get(scale, ["locality", "localadmin", "county"])
            name_match = name_match.copy()
            name_match["_subtype_rank"] = name_match["subtype"].apply(
                lambda s: preferred.index(s) if s in preferred else len(preferred)
            )
            best_div = name_match.sort_values("_subtype_rank").iloc[0]

        division_id = best_div.get("id", "")
        population_raw = best_div.get("population", None)
        best_names = best_div["_all_names"]
        best_subtype = str(best_div.get("subtype", ""))

        logger.info(
            f"DataFetcher: Matched division '{best_names[0] if best_names else location}' "
            f"(subtype={best_subtype}, id={division_id}, population={population_raw})"
        )

        # ------------------------------------------------------------------ #
        # Step 3: Query 'division_area' theme; match by division_id          #
        # ------------------------------------------------------------------ #
        try:
            area_reader = overturemaps.record_batch_reader("division_area", bbox=bbox)
            if area_reader is None:
                raise DataFetchError("Overture division_area reader returned None")
            area_table = area_reader.read_all()
        except Exception as exc:
            raise DataFetchError(f"Overture division_area query failed: {exc}") from exc

        if area_table.num_rows == 0:
            raise DataFetchError(f"Overture division_area: no results near '{location}'")

        area_df = area_table.to_pandas()

        matched = None

        # Primary match: by division_id (most reliable)
        if division_id and "division_id" in area_df.columns:
            id_match = area_df[area_df["division_id"] == division_id]
            if not id_match.empty:
                matched = id_match

        # Fallback match: by name substring
        if matched is None or len(matched) == 0:
            area_df["_area_name"] = area_df["names"].apply(
                lambda n: n.get("primary", "") if isinstance(n, dict) else ""
            )
            name_area_match = area_df[
                area_df["_area_name"].str.lower().str.contains(
                    primary_query, na=False, regex=False
                )
            ]
            if name_area_match.empty:
                raise DataFetchError(
                    f"Overture division_area: no polygon found for '{location}' "
                    f"(division_id={division_id})"
                )
            matched = name_area_match

        # ------------------------------------------------------------------ #
        # Step 4: Extract geometry and assemble result GeoDataFrame          #
        # ------------------------------------------------------------------ #
        try:
            geom = wkb.loads(matched.iloc[0]["geometry"])
        except Exception as exc:
            raise DataFetchError(
                f"Overture division_area: could not decode geometry for '{location}': {exc}"
            ) from exc

        props = {
            "name": best_names[0] if best_names else location,
            "location_query": location,
            "source": "overture_division",
            "subtype": best_subtype,
            "admin_level": str(_OVERTURE_SUBTYPE_ADMIN_LEVEL.get(best_subtype, "")),
            "population": str(int(population_raw)) if population_raw else "",
            "division_id": str(division_id),
        }
        logger.info(
            f"DataFetcher: Boundary for '{location}' obtained via Overture "
            f"(subtype={best_subtype}, name='{props['name']}', "
            f"population={props['population'] or 'unknown'})"
        )
        return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")

    def fetch_pois(
        self,
        boundary_gdf: gpd.GeoDataFrame,
        category: str,
    ) -> gpd.GeoDataFrame:
        """
        Fetch Points of Interest (POIs) within *boundary_gdf*.

        Attempts to use Overture Maps (via cloud-native GeoParquet) first
        for speed and reliability. Falls back to OSMnx
        (ox.features_from_polygon) if Overture is unavailable or fails.

        Args:
            boundary_gdf: GeoDataFrame with at least one geometry row
                          representing the region of interest.
            category: One of the keys in :data:`OVERTURE_CATEGORIES` or :data:`_OSMNX_POI_TAGS`.

        Returns:
            GeoDataFrame of POI points with columns ``name``, ``amenity``,
            ``geometry``. CRS is EPSG:4326. May be empty if no POIs found.
        """
        # Get bounding box for the query
        bounds = boundary_gdf.to_crs("EPSG:4326").total_bounds  # minx, miny, maxx, maxy
        bbox = (bounds[0], bounds[1], bounds[2], bounds[3])

        # --- Tier 1: Overture Maps ----------------------------------------
        if _OVERTURE_AVAILABLE and category in OVERTURE_CATEGORIES:
            try:
                pois_gdf = self._fetch_pois_via_overture(bbox, category)
                if not pois_gdf.empty:
                    # Clip to actual boundary polygon
                    pois_gdf = self._clip_to_boundary(pois_gdf, boundary_gdf)
                    logger.info(
                        f"DataFetcher: {len(pois_gdf)} '{category}' POIs obtained via Overture Maps"
                    )
                    return pois_gdf
            except Exception as exc:
                logger.warning(f"Overture POI fetch failed for '{category}': {exc}. Falling back to OSMnx.")

        # --- Tier 2: OSMnx features_from_polygon --------------------------
        try:
            pois_gdf = self._fetch_pois_via_osmnx(boundary_gdf, category)
            return pois_gdf
        except Exception as exc:
            raise DataFetchError(
                f"All POI retrieval methods failed for category '{category}'. Last error: {exc}"
            )

    def _fetch_pois_via_overture(self, bbox: tuple, category: str) -> gpd.GeoDataFrame:
        """Fetch POIs from Overture Maps 'place' theme."""
        import shapely.wkb as wkb

        overture_cats = OVERTURE_CATEGORIES.get(category, [])
        if not overture_cats:
            return gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")

        logger.info(f"DataFetcher: Fetching Overture POIs for '{category}' in bbox={bbox}")
        
        try:
            reader = overturemaps.record_batch_reader("place", bbox=bbox)
            if reader is None:
                return gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")
            
            table = reader.read_all()
        except Exception as exc:
            logger.warning(f"Overture reader error: {exc}")
            return gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")

        if table.num_rows == 0:
            return gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")

        # In Overture 'place' schema:
        # - names is a struct { primary: string, ... }
        # - categories is a struct { primary: string, ... }
        # - geometry is binary (WKB)
        
        # We need to filter by category. Overture categories are structured.
        # We'll check if categories.primary is in our list.
        cat_column = table.column("categories")
        # Flatten struct to access 'primary'
        # In newer pyarrow, we can use field access
        try:
            # Use pyarrow compute to access nested 'primary' field in the 'categories' struct column
            primary_categories = pc.struct_field(table.column("categories"), "primary")
            # Filter rows where the primary category matches our mapped list
            mask = pc.is_in(primary_categories, value_set=pa.array(overture_cats))
            filtered_table = table.filter(mask)
        except Exception as filter_exc:
            logger.warning(f"Overture filtering error: {filter_exc}. Falling back to Pandas-based filter.")
            # Fallback to pandas filtering if pyarrow compute fails
            df = table.to_pandas()
            df['primary_cat'] = df['categories'].apply(lambda x: x.get('primary') if isinstance(x, dict) else None)
            df = df[df['primary_cat'].isin(overture_cats)]
            filtered_table = pa.Table.from_pandas(df)

        if filtered_table.num_rows == 0:
            return gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")

        # Convert to GeoDataFrame
        df = filtered_table.to_pandas()
        
        # Extract name and amenity
        df['name'] = df['names'].apply(lambda x: x.get('primary', '') if isinstance(x, dict) else '')
        df['amenity'] = df['categories'].apply(lambda x: x.get('primary', category) if isinstance(x, dict) else category)
        
        # Load geometries
        geoms = [wkb.loads(g) for g in df['geometry']]
        
        pois_gdf = gpd.GeoDataFrame(
            df[['name', 'amenity']],
            geometry=geoms,
            crs="EPSG:4326"
        )
        return pois_gdf.reset_index(drop=True)

    def _fetch_pois_via_osmnx(
        self,
        boundary_gdf: gpd.GeoDataFrame,
        category: str,
    ) -> gpd.GeoDataFrame:
        """Fetch POIs inside *boundary_gdf* via osmnx.features_from_polygon.

        osmnx handles Overpass requests (with caching + rate-limit + retries)
        internally, so we don't maintain any Overpass mirror logic here.
        """
        if category not in _OSMNX_POI_TAGS:
            raise DataFetchError(
                f"Unknown POI category '{category}'. "
                f"Supported: {sorted(_OSMNX_POI_TAGS.keys())}"
            )

        ox = self._configure_osmnx_once()
        tags = _OSMNX_POI_TAGS[category]

        polygon = unary_union(boundary_gdf.to_crs("EPSG:4326").geometry.values)
        if polygon.is_empty:
            return gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")

        with timed("pois.fetch", source="OSMnx", detail=category) as t:
            try:
                feats = ox.features_from_polygon(polygon, tags=tags)
            except Exception as exc:
                raise DataFetchError(
                    f"OSMnx POI fetch failed for '{category}': {exc}"
                ) from exc

            if feats is None or len(feats) == 0:
                t.detail = f"{category} → 0"
                return gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")

            feats = feats.to_crs("EPSG:4326").reset_index()
            # osmnx returns mixed Point/Polygon/LineString geometries; reduce
            # polygons/lines to their centroid so downstream solvers get points.
            feats["geometry"] = feats.geometry.apply(
                lambda g: g if g.geom_type == "Point" else g.centroid
            )
            name_col = "name" if "name" in feats.columns else None
            amen_col = "amenity" if "amenity" in feats.columns else None
            shop_col = "shop" if "shop" in feats.columns else None
            rows = {
                "name": feats[name_col].fillna("").astype(str) if name_col else "",
                "amenity": (
                    feats[amen_col].fillna(feats[shop_col] if shop_col else "").astype(str)
                    if amen_col else (feats[shop_col].astype(str) if shop_col else category)
                ),
            }
            pois_gdf = gpd.GeoDataFrame(rows, geometry=feats.geometry.values, crs="EPSG:4326")
            pois_gdf = self._clip_to_boundary(pois_gdf, boundary_gdf)
            t.detail = f"{category} → {len(pois_gdf)}"
            return pois_gdf

    def _clip_to_boundary(self, gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clip a GeoDataFrame to the actual boundary polygon."""
        try:
            boundary_union = unary_union(
                boundary_gdf.to_crs("EPSG:4326").geometry.values
            )
            clipped = gdf[gdf.geometry.within(boundary_union)].copy()
            logger.info(f"DataFetcher: Clipped {len(gdf)} points to {len(clipped)} via boundary polygon")
            return clipped.reset_index(drop=True)
        except Exception as clip_err:
            logger.warning(f"Could not clip to boundary polygon: {clip_err}")
            return gdf

    def fetch_population(
        self,
        boundary_gdf: gpd.GeoDataFrame,
        n_points: Optional[int] = None,
        random_seed: int = 42,
    ) -> gpd.GeoDataFrame:
        """
        Generate a population demand grid within *boundary_gdf*.

        Uses rejection sampling to place *n_points* random Point geometries
        strictly inside the boundary polygon.  Each point is assigned a
        population weight derived from a real total-population estimate.

        Population count resolution:
        - When *n_points* is ``None`` (default), the count is derived from the
          boundary area using ``compute_n_points_from_area()``:
          ``n = clamp(sqrt(area_km²) × 8, 50, 2000)``.
        - Pass an explicit integer to override.

        Total population:
        - Uses real population from Overture / OSM metadata on *boundary_gdf*
          if the ``population`` column is populated.
        - Falls back to area × 250 people/km² density estimate.
        - Last resort: _DEFAULT_TOTAL_POPULATION constant.

        Prior to generating synthetic data, this method attempts to fetch
        real population estimates from the Humanitarian Data Exchange (HDX)
        Facebook high-resolution maps. The synthetic fallback is used when
        real-data sources fail or are too large to process in memory safely.

        Args:
            boundary_gdf: Region of interest (any CRS; reprojected internally).
            n_points:     Demand-point count. ``None`` = auto-compute from area.
            random_seed:  Seed for the internal RNG.

        Returns:
            GeoDataFrame with ``population`` column and Point geometry.
            CRS is EPSG:4326.

        Raises:
            PopulationDataError: If the boundary polygon is invalid or empty.
        """
        # --- Auto-compute n_points from boundary area when not specified ----
        if n_points is None:
            try:
                from utils.scale_classifier import compute_n_points_from_area
                area_km2 = boundary_gdf.to_crs("EPSG:6933").geometry.area.sum() / 1e6
                n_points = compute_n_points_from_area(area_km2)
                logger.info(
                    f"DataFetcher: Auto n_points={n_points} "
                    f"from boundary area={area_km2:.1f} km²"
                )
            except Exception as exc:
                logger.warning(f"Could not auto-compute n_points from area: {exc}. Using 200.")
                n_points = 200

        # --- Tier 1: Try to fetch real population data from HDX open API ---
        try:
            pop_gdf = self._fetch_population_hdx(boundary_gdf)
            if pop_gdf is not None and len(pop_gdf) > 0:
                logger.info(
                    f"DataFetcher: Using real HDX population data: {len(pop_gdf)} cells"
                )
                return pop_gdf
        except Exception as exc:
            logger.warning(
                f"HDX population fetch failed: {exc}. Falling back to synthetic grid."
            )

        # --- Tier 2: Synthetic grid (rejection-sampled strictly inside boundary) ---
        logger.info(
            f"DataFetcher: Generating synthetic population grid "
            f"({n_points} points) within boundary (seed={random_seed})"
        )

        try:
            boundary_union = unary_union(
                boundary_gdf.to_crs("EPSG:4326").geometry.values
            )
        except Exception as exc:
            raise PopulationDataError(
                f"Cannot compute boundary union for population grid: {exc}"
            ) from exc

        if boundary_union is None or boundary_union.is_empty:
            raise PopulationDataError("Boundary geometry is empty.")

        # Detect if boundary is a plain bbox rectangle (Photon fallback)
        # and warn the user that points near coasts may fall in water.
        bounds = boundary_union.bounds
        bbox_geom = box(*bounds)
        is_plain_bbox = boundary_union.equals(bbox_geom) or (
            abs(boundary_union.area - bbox_geom.area) / max(bbox_geom.area, 1e-10) < 1e-6
        )
        if is_plain_bbox:
            logger.warning(
                "DataFetcher: Boundary polygon is a plain bounding-box rectangle. "
                "Some population points may fall in water bodies. "
                "Consider fetching a proper administrative boundary polygon."
            )

        minx, miny, maxx, maxy = bounds

        points: list[Point] = []
        MAX_ATTEMPTS = n_points * 50  # extra headroom for coastal areas
        attempts = 0
        rng = random.Random(random_seed)

        while len(points) < n_points and attempts < MAX_ATTEMPTS:
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            pt = Point(x, y)
            if boundary_union.contains(pt):
                points.append(pt)
            attempts += 1

        if not points:
            raise PopulationDataError(
                "Could not generate any population points within boundary. "
                "The boundary polygon may be too small or degenerate."
            )

        if len(points) < n_points:
            logger.warning(
                f"Could only generate {len(points)}/{n_points} population points "
                f"after {MAX_ATTEMPTS} attempts (thin polygon?)."
            )

        # Use a real population estimate so per-point weights are meaningful.
        total_pop = self._estimate_total_population(boundary_gdf)
        pop_per_point = total_pop / len(points)

        gdf = gpd.GeoDataFrame(
            {"population": [pop_per_point] * len(points)},
            geometry=points,
            crs="EPSG:4326",
        )
        gdf["data_source"] = "synthetic_uniform_grid"
        logger.info(
            f"DataFetcher: Generated {len(gdf)} synthetic population points "
            f"({pop_per_point:.0f} pop each, total={total_pop:,.0f})"
        )
        return gdf

    def _estimate_total_population(self, boundary_gdf: gpd.GeoDataFrame) -> int:
        """
        Derive a real population estimate for *boundary_gdf*.

        Sources tried in order:
        1. ``population`` column on *boundary_gdf* — populated by Overture
           ``division`` theme or Nominatim ``extratags``.
        2. Area × 250 people/km² (rough global average density).
        3. :data:`_DEFAULT_TOTAL_POPULATION` constant (last resort).
        """
        # Source 1: metadata from Overture division or Nominatim extratags
        if "population" in boundary_gdf.columns:
            raw = str(boundary_gdf["population"].iloc[0] or "").replace(",", "").strip()
            try:
                pop = int(float(raw))
                if pop > 0:
                    logger.info(f"DataFetcher: Using metadata population: {pop:,}")
                    return pop
            except (ValueError, TypeError):
                pass

        # Source 2: area-based density estimate
        try:
            area_km2 = boundary_gdf.to_crs("EPSG:6933").geometry.area.sum() / 1e6
            estimated = int(area_km2 * 250)
            if estimated > 0:
                logger.info(
                    f"DataFetcher: Estimated population from area "
                    f"{area_km2:.0f} km² × 250/km² = {estimated:,}"
                )
                return estimated
        except Exception as exc:
            logger.debug(f"Area-based population estimate failed: {exc}")

        return _DEFAULT_TOTAL_POPULATION

    def _fetch_population_hdx(self, boundary_gdf: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
        """
        Attempt to fetch population data from Humanitarian Data Exchange (HDX) Facebook Maps.
        Uses the hdx-python-api to find the country dataset and process its CSV resource.
        Returns None (rather than raising) if data is unavailable or file is too large.
        """
        import unicodedata
        import zipfile
        import os
        
        try:
            from hdx.utilities.easy_logging import setup_logging
            from hdx.api.configuration import Configuration
            from hdx.data.dataset import Dataset
        except ImportError:
            logger.warning("hdx-python-api is not installed. Population fetching from HDX will be skipped.")
            return None
        
        try:
            boundary_4326 = boundary_gdf.to_crs("EPSG:4326")
            boundary_union = unary_union(boundary_4326.geometry.values)
            if boundary_union is None or boundary_union.is_empty:
                return None

            minx, miny, maxx, maxy = boundary_union.bounds
            centroid = boundary_union.centroid

            # 1. Determine country name — prefer metadata already on the boundary GDF,
            #    fall back to Photon reverse geocoding on the centroid.
            country_raw = ""

            # Check boundary GDF for a pre-populated country field
            for col in ("country", "country_code", "country_name"):
                if col in boundary_4326.columns:
                    val = str(boundary_4326[col].iloc[0] or "").strip()
                    if val:
                        country_raw = val
                        break

            # Also try the location_query column (e.g. "Bahawalpur, Pakistan")
            if not country_raw and "location_query" in boundary_4326.columns:
                lq = str(boundary_4326["location_query"].iloc[0] or "")
                parts = [p.strip() for p in lq.split(",")]
                if len(parts) >= 2:
                    country_raw = parts[-1]  # last part is usually the country

            # Final fallback: Photon /reverse on the centroid
            if not country_raw:
                reverse_url = PHOTON_URL + "/reverse"
                rev_params = {"lat": centroid.y, "lon": centroid.x}
                try:
                    rev_resp = self._make_request(reverse_url, params=rev_params, timeout=20)
                    rev_data = rev_resp.json()
                    features = rev_data.get("features", [])
                    if features:
                        feat = features[0]
                        feat_coords = feat.get("geometry", {}).get("coordinates", [])
                        if feat_coords and len(feat_coords) >= 2:
                            feat_lon, feat_lat = float(feat_coords[0]), float(feat_coords[1])
                            # Reject if the returned point is implausibly far from our centroid
                            if abs(feat_lon - centroid.x) < 15 and abs(feat_lat - centroid.y) < 15:
                                country_raw = feat.get("properties", {}).get("country", "")
                            else:
                                logger.warning(
                                    f"Photon reverse geocode returned a location far from centroid "
                                    f"({feat_lat:.2f},{feat_lon:.2f} vs "
                                    f"{centroid.y:.2f},{centroid.x:.2f}); ignoring."
                                )
                        else:
                            country_raw = feat.get("properties", {}).get("country", "")
                except Exception as e:
                    logger.debug(f"Reverse geocode failed: {e}")

            if not country_raw:
                logger.warning("Could not determine country for HDX population fetch.")
                return None

            logger.info(f"HDX population: resolved country as '{country_raw}'")

            # Normalize country name (e.g., 'Perú' -> 'peru', 'United Kingdom' -> 'united-kingdom')
            country_norm = unicodedata.normalize('NFKD', country_raw).encode('ASCII', 'ignore').decode('utf-8')
            country_norm = country_norm.lower().replace(" ", "-")

            # 2. Search HDX for population density dataset
            try:
                Configuration.create(hdx_site="prod", user_agent="SOCA_spopt_agent", hdx_read_only=True)
            except Exception:
                pass  # configuration already set up
            
            # 2. Search HDX — try multiple query patterns in priority order
            _SIZE_LIMIT = 60 * 1024 * 1024  # 60 MB
            _SKIP_KEYWORDS = ["children", "elderly", "youth", "men", "women",
                              "under_five", "reproductive", "15_24", "60_plus"]
            _SKIP_FORMATS = {"geotiff", "tif", "tiff"}

            def _pick_resource(ds):
                """Return (resource, kind) where kind is 'csv' or 'gpkg', or None."""
                resources = ds.get_resources()
                # Priority 1: general-population CSV < size limit
                for r in resources:
                    fmt = r.get_format().lower()
                    name = r.get("name", "").lower()
                    if any(x in fmt for x in ("csv",)) or name.endswith((".csv", ".csv.zip")):
                        if any(kw in name for kw in _SKIP_KEYWORDS):
                            continue
                        if "part_1" in name:
                            continue
                        sz = r.get("size", 0) or 0
                        if sz > _SIZE_LIMIT:
                            continue
                        if "general" in name or "overall" in name or "population_" in name:
                            return r, "csv"
                # Priority 2: GeoPackage (Kontur-style) < size limit
                for r in resources:
                    fmt = r.get_format().lower()
                    name = r.get("name", "").lower()
                    if "gpkg" in fmt or name.endswith(".gpkg"):
                        if any(kw in name for kw in _SKIP_KEYWORDS):
                            continue
                        sz = r.get("size", 0) or 0
                        if sz > _SIZE_LIMIT:
                            continue
                        return r, "gpkg"
                return None, None

            def _dataset_score(ds):
                """Higher score = more suitable for demand-point generation."""
                title = ds.get("title", "").lower()
                if "population density" in title or "h3 hexagon" in title:
                    return 3
                if "hrsl" in title or "high resolution population" in title:
                    return 2
                if "population" in title and "administrative" not in title and "boundary" not in title:
                    return 1
                return 0

            # Collect all candidate (dataset, resource, kind) tuples across queries
            # Note: Facebook/Meta HR population maps are excluded — discontinued since 2024.
            all_candidates: list[tuple] = []
            for query in [
                f"{country_norm} kontur population",
                f"{country_norm} hrsl",
                f"{country_norm} population density",
            ]:
                try:
                    found = Dataset.search_in_hdx(query, rows=5)
                except Exception as search_exc:
                    logger.warning(f"HDX search failed for query '{query}': {search_exc}")
                    continue
                for ds in found:
                    r, kind = _pick_resource(ds)
                    if r is not None:
                        all_candidates.append((_dataset_score(ds), ds, r, kind))

            if not all_candidates:
                logger.warning(f"No usable HDX population resource found for '{country_norm}'.")
                return None

            # Pick the highest-scored candidate
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            _, best_ds, target_resource, resource_kind = all_candidates[0]
            datasets = [best_ds]

            if not target_resource:
                logger.warning(f"No usable HDX population resource found for '{country_norm}'.")
                return None

            logger.info(
                f"HDX: downloading '{target_resource.get('name')}' "
                f"({resource_kind}, {(target_resource.get('size') or 0)//1024//1024}MB) "
                f"from '{datasets[0].get('title')}'"
            )

            # 3. Download the resource
            # hdx-python-api versions differ: some return (url, path), others return
            # just a path. Handle both and always convert to a plain string.
            try:
                dl_result = target_resource.download()
                if isinstance(dl_result, (list, tuple)) and len(dl_result) == 2:
                    _url, path = dl_result
                else:
                    path = dl_result  # newer API returns path directly
            except Exception as e:
                logger.error(f"Failed to download HDX resource: {e}")
                return None

            if not path:
                logger.error("HDX resource download returned no path.")
                return None

            # Always convert to str — Path objects cause 'has no attribute endswith'
            path_str = str(path)

            # 4a. CSV handler
            if resource_kind == "csv":
                try:
                    if path_str.endswith('.zip') or zipfile.is_zipfile(path_str):
                        df = pd.read_csv(path_str, compression='zip')
                    else:
                        df = pd.read_csv(path_str)
                except Exception as e:
                    logger.error(f"Failed to read HDX CSV: {e}")
                    return None
                finally:
                    if os.path.exists(path_str):
                        try:
                            os.unlink(path_str)
                        except Exception:
                            pass

                df.columns = [str(c).lower().strip() for c in df.columns]
                if "lon" in df.columns and "longitude" not in df.columns:
                    df.rename(columns={"lon": "longitude"}, inplace=True)
                if "lat" in df.columns and "latitude" not in df.columns:
                    df.rename(columns={"lat": "latitude"}, inplace=True)

                if "longitude" not in df.columns or "latitude" not in df.columns:
                    logger.error("HDX CSV missing latitude/longitude columns.")
                    return None

                df_filtered = df[
                    (df["longitude"] >= minx) & (df["longitude"] <= maxx) &
                    (df["latitude"] >= miny) & (df["latitude"] <= maxy)
                ].copy()

                if df_filtered.empty:
                    logger.warning("HDX CSV has no rows within boundary bbox.")
                    return None

                pop_col = next((c for c in df_filtered.columns if "population_" in c), None)
                if not pop_col:
                    pop_col = "population" if "population" in df_filtered.columns else df_filtered.columns[-1]

                rows = []
                for _, row in df_filtered.iterrows():
                    try:
                        pt = Point(row["longitude"], row["latitude"])
                        if boundary_union.contains(pt):
                            pop_val = float(row[pop_col])
                            if pop_val > 0:
                                rows.append({"population": pop_val, "geometry": pt})
                    except Exception:
                        continue

                if not rows:
                    logger.warning("HDX CSV: no points fell inside boundary polygon.")
                    return None

                gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
                gdf["data_source"] = "hdx_facebook_population"
                return gdf

            # 4b. GeoPackage handler (Kontur H3 hexagons)
            elif resource_kind == "gpkg":
                import gzip as _gzip, shutil
                # HDX may download as gzip-compressed GPKG (magic bytes \x1f\x8b).
                # Decompress and ensure a clean .gpkg extension for pyogrio.
                gpkg_path = path_str
                extra_files = set()
                try:
                    with open(path_str, "rb") as _f:
                        is_gzip = _f.read(2) == b"\x1f\x8b"
                    if is_gzip:
                        gpkg_path = path_str + "_decompressed.gpkg"
                        extra_files.add(gpkg_path)
                        with _gzip.open(path_str, "rb") as _gz, open(gpkg_path, "wb") as _out:
                            shutil.copyfileobj(_gz, _out)
                    elif not path_str.endswith(".gpkg"):
                        gpkg_path = path_str + ".gpkg"
                        extra_files.add(gpkg_path)
                        shutil.copy(path_str, gpkg_path)
                except Exception as prep_exc:
                    logger.warning(f"HDX GeoPackage pre-processing failed: {prep_exc}")

                try:
                    pop_gdf_raw = gpd.read_file(gpkg_path)
                except Exception as e:
                    logger.error(f"Failed to read HDX GeoPackage: {e}")
                    return None
                finally:
                    for p in ({path_str} | extra_files):
                        if os.path.exists(p):
                            try:
                                os.unlink(p)
                            except Exception:
                                pass

                pop_gdf_raw = pop_gdf_raw.to_crs("EPSG:4326")

                # Clip to boundary bbox first, then intersect
                pop_gdf_raw = pop_gdf_raw.cx[minx:maxx, miny:maxy]
                if pop_gdf_raw.empty:
                    logger.warning("HDX GeoPackage has no features within boundary bbox.")
                    return None

                clipped = gpd.clip(pop_gdf_raw, boundary_union)
                if clipped.empty:
                    logger.warning("HDX GeoPackage: no features intersect boundary polygon.")
                    return None

                # Kontur uses 'population' column; centroids become demand points
                pop_col = next(
                    (c for c in clipped.columns if "population" in c.lower()),
                    None
                )
                if not pop_col:
                    logger.warning("HDX GeoPackage has no recognisable population column.")
                    return None

                clipped = clipped[clipped[pop_col] > 0].copy()
                # Reproject to a metric CRS for accurate centroids, then back to WGS-84
                clipped["geometry"] = clipped.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326")
                clipped = clipped.rename(columns={pop_col: "population"})
                result = clipped[["population", "geometry"]].copy()
                result["data_source"] = "hdx_kontur_population"
                return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")

            return None

        except Exception as exc:
            logger.warning(f"HDX population fetch failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nominatim_rate_limit(self) -> None:
        """Enforce >= 1 second between Nominatim requests (ToS)."""
        elapsed = time.monotonic() - self._last_nominatim_call
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_nominatim_call = time.monotonic()

    def _make_request(
        self,
        url: str,
        params: Optional[dict] = None,
        method: str = "GET",
        timeout: int = 30,
    ) -> "requests.Response":
        """
        Send an HTTP request with retry logic.

        Retries up to :data:`_MAX_RETRIES` times with exponential back-off
        on transient errors (timeouts, 5xx responses).

        Args:
            url: Full URL to request.
            params: Query params (GET) or form data (POST).
            method: ``"GET"`` or ``"POST"``.
            timeout: Per-attempt timeout in seconds.

        Returns:
            Successful :class:`requests.Response`.

        Raises:
            DataFetchError: After all retries are exhausted.
        """
        headers = {
            "User-Agent": "SOCA/1.0 (Spatial Optimization Conversational Agent; "
                          "academic research; contact: soca@example.com)",
            "Accept": "application/json",
        }

        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            delay = _RETRY_BASE_DELAY * (2 ** attempt)  # 1, 2, 4 seconds
            try:
                if method.upper() == "POST":
                    resp = requests.post(
                        url, data=params, headers=headers, timeout=timeout
                    )
                else:
                    resp = requests.get(
                        url, params=params, headers=headers, timeout=timeout
                    )

                # Raise for 4xx/5xx — but NOT for 429 (rate limit) which we retry
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", delay))
                    logger.warning(
                        f"Rate limited by {url}; sleeping {retry_after}s "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                    )
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                return resp

            except requests.exceptions.Timeout as exc:
                last_exc = exc
                logger.warning(
                    f"Timeout on attempt {attempt + 1}/{_MAX_RETRIES} to {url}; "
                    f"retrying in {delay}s"
                )
                time.sleep(delay)

            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                logger.warning(
                    f"Connection error on attempt {attempt + 1}/{_MAX_RETRIES} to {url}; "
                    f"retrying in {delay}s"
                )
                time.sleep(delay)

            except requests.exceptions.HTTPError as exc:
                # 5xx → retry; 4xx (other than 429) → fail immediately
                status = exc.response.status_code if exc.response is not None else 0
                if status >= 500:
                    last_exc = exc
                    logger.warning(
                        f"HTTP {status} on attempt {attempt + 1}/{_MAX_RETRIES} to {url}; "
                        f"retrying in {delay}s"
                    )
                    time.sleep(delay)
                else:
                    raise DataFetchError(
                        f"HTTP {status} error from {url}: {exc}"
                    ) from exc

            except Exception as exc:
                raise DataFetchError(
                    f"Unexpected error requesting {url}: {exc}"
                ) from exc

        raise DataFetchError(
            f"All {_MAX_RETRIES} attempts failed for {url}. "
            f"Last error: {last_exc}"
        )
