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

# Shared User-Agent — Nominatim ToS requires an app-specific UA and actively
# blocks requests whose UA contains placeholder domains ("example.com",
# "example.org") or library defaults. We therefore identify this app by its
# public repository URL, which Nominatim's policy accepts as valid contact
# information. Mirrors utils/geocoder.py:_USER_AGENT so both endpoints
# identify as the same SOCA app. Also handed to osmnx via
# ox.settings.http_user_agent.
_USER_AGENT = (
    "SOCA-spopt/1.0 (Spatial Optimization Conversational Agent; "
    "academic research; +https://github.com/soca-spopt/soca)"
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

        Fallback chain, ordered by *measured* round-trip time (fastest first)
        so users are never left waiting for a slow tier when a fast one works:

           1. Nominatim /lookup    — direct OSM id (when hint supplied); ~0.5 s
           2. Nominatim /search    — polygon_geojson=1; ~1–2 s, ToS-compliant
           3. Overture division    — slow cloud-native GeoParquet; 30 s–3 min
           4. Photon bbox          — rectangular fallback; ~1 s
           5. GADM                 — last-resort country/state admin boundaries

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

        last_exc: Optional[Exception] = None

        # --- Tier 1: Direct OSM id lookup via Nominatim /lookup -----------
        # When the caller has already disambiguated the place (e.g. from the
        # AOI picker's autocomplete) we know the exact relation/way/node id.
        # Nominatim /lookup returns a polygon in one round-trip (<1 s) — the
        # fastest path by far. We call it directly with our compliant UA so
        # we don't depend on osmnx's User-Agent handling.
        hint_osm_id = getattr(hint, "osm_id", None) if hint is not None else None
        hint_osm_type = getattr(hint, "osm_type", None) if hint is not None else None
        if hint_osm_id and hint_osm_type in ("R", "W", "N"):
            try:
                with timed(
                    "boundary.fetch", source="Nominatim/id",
                    detail=f"{hint_osm_type}{hint_osm_id}",
                ) as t:
                    gdf = self._fetch_boundary_via_nominatim_lookup(
                        osm_type=hint_osm_type, osm_id=int(hint_osm_id),
                        location=location,
                    )
                    t.detail = f"{hint_osm_type}{hint_osm_id} · {gdf.geometry.iloc[0].geom_type}"
                return gdf
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"Nominatim id lookup failed for {hint_osm_type}{hint_osm_id}: {exc}. "
                    "Falling back to full tier chain."
                )

        # --- Tier 2: Nominatim /search with polygon_geojson=1 -------------
        try:
            with timed("boundary.fetch", source="Nominatim", detail=location) as t:
                gdf = self._fetch_boundary_via_nominatim(location)
                t.detail = f'{location} · {gdf.geometry.iloc[0].geom_type}'
            return gdf
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"Nominatim boundary fetch failed for '{location}': {exc}. "
                "Falling back to Overture."
            )

        # --- Tier 3: Overture Maps division + division_area (slow) --------
        if _OVERTURE_AVAILABLE:
            try:
                with timed("boundary.fetch", source="Overture", detail=location) as t:
                    gdf = self._fetch_boundary_via_overture(
                        location, admin_level=admin_level, scale=scale
                    )
                    t.detail = f'{location} · {gdf.geometry.iloc[0].geom_type}'
                return gdf
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"Overture boundary fetch failed for '{location}': {exc}. "
                    "Falling back to Photon."
                )

        # --- Tier 4: Photon bbox (rectangular fallback) -------------------
        try:
            with timed("boundary.fetch", source="Photon", detail=location) as t:
                gdf = self._fetch_boundary_via_photon(location)
                src = gdf.iloc[0].get("source", "photon") if len(gdf) else "photon"
                t.detail = f'{location} · {gdf.geometry.iloc[0].geom_type} · via {src}'
            return gdf
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"Photon bbox fetch failed for '{location}': {exc}. Falling back to GADM."
            )

        # --- Tier 5: GADM (last resort, country / state only) -------------
        if _GADM_AVAILABLE:
            try:
                with timed("boundary.fetch", source="GADM", detail=location) as t:
                    gdf = self._fetch_boundary_via_gadm(location, admin_level=admin_level)
                    t.detail = f'{location} · {gdf.geometry.iloc[0].geom_type}'
                return gdf
            except Exception as exc:
                last_exc = exc

        raise GeocodingError(
            f"All geocoding backends failed for '{location}'. "
            f"Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Boundary backend implementations
    # ------------------------------------------------------------------

    def _configure_osmnx_once(self):
        """Idempotent osmnx.settings configuration. Returns the osmnx module.

        Used by the POI fetch tier only — boundary lookups now hit Nominatim
        directly for predictable latency and UA control.
        """
        import osmnx as ox
        if getattr(self, "_osmnx_configured", False):
            return ox
        ox.settings.use_cache = True
        ox.settings.requests_timeout = 180
        ox.settings.overpass_rate_limit = True
        ox.settings.log_console = False
        ox.settings.http_user_agent = _USER_AGENT
        self._osmnx_configured = True
        return ox

    def _fetch_boundary_via_nominatim_lookup(
        self,
        *,
        osm_type: str,
        osm_id: int,
        location: str,
    ) -> gpd.GeoDataFrame:
        """Fetch a polygon from Nominatim /lookup by prefixed OSM id.

        The /lookup endpoint resolves one or more OSM ids to full objects in
        a single round-trip — the fastest boundary path available. We request
        polygon_geojson=1 so the response already contains the geometry.
        """
        type_map = {"R": "R", "W": "W", "N": "N",
                    "relation": "R", "way": "W", "node": "N"}
        t_prefix = type_map.get(osm_type)
        if t_prefix is None:
            raise GeocodingError(
                f"Invalid osm_type '{osm_type}' (expected one of R/W/N)."
            )

        self._nominatim_rate_limit()
        params = {
            "osm_ids": f"{t_prefix}{int(osm_id)}",
            "format": "geojson",
            "polygon_geojson": 1,
            "addressdetails": 1,
            "extratags": 1,
        }
        try:
            response = self._make_request(
                NOMINATIM_URL + "/lookup", params=params, timeout=20
            )
        except DataFetchError as exc:
            raise GeocodingError(
                f"Network error in Nominatim lookup for {t_prefix}{osm_id}: {exc}"
            ) from exc

        try:
            data = response.json()
        except Exception as exc:
            raise GeocodingError(
                f"Invalid JSON from Nominatim lookup: {exc}"
            ) from exc

        features = data.get("features", [])
        if not features:
            raise GeocodingError(
                f"Nominatim /lookup returned no feature for {t_prefix}{osm_id}."
            )

        feature = features[0]
        try:
            geom = shape(feature["geometry"])
        except Exception as exc:
            raise GeocodingError(
                f"Could not parse geometry from Nominatim /lookup: {exc}"
            ) from exc

        if geom.is_empty:
            raise GeocodingError(
                f"Nominatim /lookup returned empty geometry for {t_prefix}{osm_id}."
            )
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise GeocodingError(
                f"Nominatim /lookup returned non-polygon geometry "
                f"({geom.geom_type}) for {t_prefix}{osm_id}."
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
        """Fetch boundary polygon via Nominatim /search with polygon_geojson=1.

        Primary boundary backend — returns proper admin polygons in 1–2 s.
        Also captures country + country_code from addressdetails and any
        population tag from extratags, so downstream HDX lookups don't need
        a second round-trip.
        """
        self._nominatim_rate_limit()

        params = {
            "q": location,
            "format": "geojson",
            "polygon_geojson": 1,
            "limit": 3,
            "addressdetails": 1,
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

        raw_props = feature.get("properties", {}) or {}
        address = raw_props.get("address") or {}
        extratags = raw_props.get("extratags") or {}

        # Flatten to a scalar schema for downstream tools — GeoPandas attrs
        # don't round-trip nested dicts reliably.
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
        Attempt to fetch population data from Humanitarian Data Exchange (HDX).

        Prefers Kontur Population (H3 hexagons, global, CC-BY) which has a
        per-country GeoPackage for every UN member state. Returns None (rather
        than raising) if data is unavailable or the file exceeds the size
        safety limit.

        Country resolution uses ``hdx-python-country``'s fuzzy matcher, which
        handles native scripts ("پاکستان" → PAK), accented Latin names
        ("Perú" → PER), and abbreviations. This prevents the catastrophic
        failure mode where an unrecognised country string silently causes
        HDX search to return the first global dataset (which previously was
        Lebanon's Kontur file — totally unrelated).
        """
        import zipfile
        import os

        try:
            from hdx.api.configuration import Configuration
            from hdx.data.dataset import Dataset
            from hdx.location.country import Country
        except ImportError:
            logger.warning(
                "hdx-python-api is not installed. Population fetch from HDX skipped."
            )
            return None

        try:
            boundary_4326 = boundary_gdf.to_crs("EPSG:4326")
            boundary_union = unary_union(boundary_4326.geometry.values)
            if boundary_union is None or boundary_union.is_empty:
                return None

            minx, miny, maxx, maxy = boundary_union.bounds
            centroid = boundary_union.centroid

            # ---------------------------------------------------------- #
            # Step 1: Resolve ISO3 country code                           #
            # ---------------------------------------------------------- #
            # Priority order: explicit country_code on GDF → country name
            # on GDF → location_query trailer → Photon reverse geocode.
            iso3: Optional[str] = None
            country_display: str = ""

            def _try_iso(raw: str) -> Optional[str]:
                raw = (raw or "").strip()
                if not raw:
                    return None
                # 2-letter code first (returns bare string or None)
                if len(raw) == 2:
                    try:
                        code = Country.get_iso3_from_iso2(raw.upper())
                        if code:
                            return code
                    except Exception:
                        pass
                # 3-letter code passthrough
                if len(raw) == 3 and raw.isalpha():
                    try:
                        info = Country.get_country_info_from_iso3(raw.upper())
                        if info:
                            return raw.upper()
                    except Exception:
                        pass
                # Fuzzy match on full name
                try:
                    code, matched = Country.get_iso3_country_code_fuzzy(raw)
                    if matched and code:
                        return code
                except Exception:
                    pass
                return None

            # GDF columns
            for col in ("country_code", "iso3", "iso_a3"):
                if col in boundary_4326.columns:
                    val = str(boundary_4326[col].iloc[0] or "").strip()
                    iso3 = _try_iso(val)
                    if iso3:
                        country_display = val
                        break

            if not iso3:
                for col in ("country", "country_name"):
                    if col in boundary_4326.columns:
                        val = str(boundary_4326[col].iloc[0] or "").strip()
                        iso3 = _try_iso(val)
                        if iso3:
                            country_display = val
                            break

            # location_query trailer (e.g. "Bahawalpur, Pakistan")
            if not iso3 and "location_query" in boundary_4326.columns:
                lq = str(boundary_4326["location_query"].iloc[0] or "")
                parts = [p.strip() for p in lq.split(",")]
                if len(parts) >= 2:
                    iso3 = _try_iso(parts[-1])
                    if iso3:
                        country_display = parts[-1]

            # Photon reverse geocode — request English names so HDX fuzzy
            # matcher doesn't have to handle native scripts.
            if not iso3:
                reverse_url = PHOTON_URL + "/reverse"
                rev_params = {
                    "lat": centroid.y,
                    "lon": centroid.x,
                    "lang": "en",
                }
                try:
                    rev_resp = self._make_request(reverse_url, params=rev_params, timeout=20)
                    rev_data = rev_resp.json()
                    features = rev_data.get("features", [])
                    if features:
                        feat = features[0]
                        feat_coords = feat.get("geometry", {}).get("coordinates", [])
                        if feat_coords and len(feat_coords) >= 2:
                            feat_lon, feat_lat = float(feat_coords[0]), float(feat_coords[1])
                            if abs(feat_lon - centroid.x) < 15 and abs(feat_lat - centroid.y) < 15:
                                p = feat.get("properties", {}) or {}
                                countrycode = (p.get("countrycode") or "").strip().upper()
                                country_en = (p.get("country") or "").strip()
                                iso3 = _try_iso(countrycode) or _try_iso(country_en)
                                if iso3:
                                    country_display = country_en or countrycode
                            else:
                                logger.warning(
                                    "Photon reverse returned a location far from centroid "
                                    f"({feat_lat:.2f},{feat_lon:.2f} vs "
                                    f"{centroid.y:.2f},{centroid.x:.2f}); ignoring."
                                )
                except Exception as e:
                    logger.debug(f"Reverse geocode failed: {e}")

            if not iso3:
                logger.warning(
                    "HDX: could not resolve ISO3 country code for boundary; skipping."
                )
                return None

            # Canonical English country name for log + HDX queries
            try:
                country_en = Country.get_country_name_from_iso3(iso3) or country_display or iso3
            except Exception:
                country_en = country_display or iso3

            logger.info(f"HDX population: resolved country as '{country_en}' ({iso3})")

            # ---------------------------------------------------------- #
            # Step 2: Ensure HDX is configured                            #
            # ---------------------------------------------------------- #
            try:
                Configuration.create(
                    hdx_site="prod",
                    user_agent="SOCA_spopt_agent",
                    hdx_read_only=True,
                )
            except Exception:
                pass  # already configured

            # ---------------------------------------------------------- #
            # Step 3: Find a matching dataset for *this* country.         #
            #                                                             #
            # Strategy: Kontur Maps publishes per-country H3-hexagon      #
            # GeoPackages with a predictable slug:                        #
            #     kontur-population-<english-country-name>                #
            # e.g. 'kontur-population-pakistan', 'kontur-population-peru'.#
            # A direct Dataset.read_from_hdx() is ~0.5 s and is           #
            # guaranteed to hit the right country, so it replaces the     #
            # old text-search heuristic that could pick Lebanon data for  #
            # a Pakistan boundary.                                        #
            # ---------------------------------------------------------- #
            # Size ceiling lets us skip massive country files (e.g. US ~300MB,
            # Russia ~200MB) on constrained environments while still using
            # Kontur for most of the world. Opt-in override via env var.
            import os as _os
            _SIZE_LIMIT = int(
                _os.environ.get("SOCA_HDX_MAX_RESOURCE_MB", "400")
            ) * 1024 * 1024
            _SKIP_KEYWORDS = ("children", "elderly", "youth",
                              "under_five", "reproductive", "15_24", "60_plus",
                              "indicator", "health-statistics", "statistics",
                              "indicators", "who-")

            def _resource_size_mb(r) -> int:
                try:
                    return int((r.get("size") or 0)) // 1024 // 1024
                except Exception:
                    return 0

            def _pick_resource(ds):
                """Return the best (resource, kind) for population density.

                Prefers the newest Kontur GeoPackage; falls back to CSV.
                """
                try:
                    resources = ds.get_resources()
                except Exception:
                    return None, None

                gpkg_matches = []
                csv_matches = []
                for r in resources:
                    try:
                        fmt = (r.get_format() or "").lower()
                        name = (r.get("name") or "").lower()
                    except Exception:
                        continue
                    if any(kw in name for kw in _SKIP_KEYWORDS):
                        continue
                    sz = r.get("size", 0) or 0
                    if sz > _SIZE_LIMIT:
                        continue
                    if "gpkg" in fmt or "geopackage" in fmt or name.endswith((".gpkg", ".gpkg.gz")):
                        gpkg_matches.append(r)
                    elif "csv" in fmt or name.endswith((".csv", ".csv.zip")):
                        if "part_1" in name or "part_2" in name:
                            continue
                        csv_matches.append(r)

                def _newest(rs):
                    # Sort by name (Kontur names include date e.g. _20231101)
                    return sorted(rs, key=lambda r: r.get("name") or "", reverse=True)[0] if rs else None

                r = _newest(gpkg_matches)
                if r is not None:
                    return r, "gpkg"
                r = _newest(csv_matches)
                if r is not None:
                    return r, "csv"
                return None, None

            def _dataset_matches_country(ds, iso3_upper: str) -> bool:
                """True when the dataset's HDX groups confirm this country."""
                try:
                    for g in ds.get("groups") or []:
                        g_name = (g.get("name") or "").lower().strip()
                        if not g_name:
                            continue
                        # HDX group names are ISO3 lowercase (e.g. 'pak').
                        if g_name.upper() == iso3_upper:
                            return True
                except Exception:
                    pass
                return False

            iso3_upper = iso3.upper()

            # --- Attempt 1: direct Kontur slug lookup (fastest path) ----
            # Kontur publishes per-country datasets at predictable slugs.
            country_slug = country_en.lower()
            # Normalise common punctuation that HDX strips from slugs
            for ch in [",", "'", "`", "(", ")", "."]:
                country_slug = country_slug.replace(ch, "")
            country_slug = "-".join(country_slug.split())
            direct_slug = f"kontur-population-{country_slug}"

            best_ds = None
            target_resource = None
            resource_kind = None

            try:
                ds = Dataset.read_from_hdx(direct_slug)
            except Exception as exc:
                logger.debug(f"HDX direct slug '{direct_slug}' error: {exc}")
                ds = None

            if ds is not None and _dataset_matches_country(ds, iso3_upper):
                r, kind = _pick_resource(ds)
                if r is not None:
                    best_ds = ds
                    target_resource = r
                    resource_kind = kind
                    logger.info(f"HDX: using direct Kontur slug '{direct_slug}'")

            # --- Attempt 2: search-based fallback ----------------------
            # Only accept datasets whose title marks them as spatial population
            # density products (Kontur, HRSL, WorldPop). A generic "population"
            # hit is not enough — old logic accidentally returned Lebanon data
            # for a Pakistan boundary and USA health-indicator CSVs for
            # Brooklyn.
            _SPATIAL_POP_KEYWORDS = (
                "kontur", "h3 hexagon", "high resolution population",
                "hrsl", "worldpop", "population density",
            )

            if target_resource is None:
                queries = [
                    f'"{country_en}" kontur population density',
                    f'"{country_en}" worldpop population',
                    f'"{country_en}" hrsl population',
                ]
                for query in queries:
                    try:
                        found = Dataset.search_in_hdx(query, rows=10)
                    except Exception as search_exc:
                        logger.warning(f"HDX search failed for '{query}': {search_exc}")
                        continue
                    for cand_ds in found:
                        if not _dataset_matches_country(cand_ds, iso3_upper):
                            continue
                        title_lower = (cand_ds.get("title") or "").lower()
                        if not any(kw in title_lower for kw in _SPATIAL_POP_KEYWORDS):
                            continue
                        r, kind = _pick_resource(cand_ds)
                        if r is not None:
                            best_ds = cand_ds
                            target_resource = r
                            resource_kind = kind
                            break
                    if target_resource is not None:
                        break

            if target_resource is None:
                logger.warning(
                    f"No country-matched HDX population resource found for "
                    f"'{country_en}' ({iso3_upper}); will use synthetic grid."
                )
                return None

            # 3. Download the resource — cached under ~/.cache/soca/hdx.
            # HDX's Resource.download() doesn't check if the file already
            # exists locally, so we maintain our own simple on-disk cache
            # keyed by resource id to skip re-downloading 300MB GeoPackages.
            import hashlib
            from pathlib import Path as _Path

            cache_dir = _Path(
                _os.environ.get("SOCA_HDX_CACHE_DIR")
                or _Path.home() / ".cache" / "soca" / "hdx"
            )
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                cache_dir = None  # fall back to fresh download each time

            res_name = target_resource.get("name") or "resource"
            res_id = target_resource.get("id") or hashlib.sha1(
                res_name.encode("utf-8", "ignore")
            ).hexdigest()[:16]
            cache_path: Optional[_Path] = None
            if cache_dir is not None:
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", res_name)[:80]
                cache_path = cache_dir / f"{res_id}_{safe_name}"

            path_str: Optional[str] = None
            if cache_path is not None and cache_path.exists() and cache_path.stat().st_size > 0:
                logger.info(
                    f"HDX: using cached '{res_name}' from {cache_path} "
                    f"({cache_path.stat().st_size // 1024 // 1024}MB)"
                )
                path_str = str(cache_path)
            else:
                logger.info(
                    f"HDX: downloading '{res_name}' "
                    f"({resource_kind}, {_resource_size_mb(target_resource)}MB) "
                    f"from '{best_ds.get('title')}'"
                )
                try:
                    dl_result = target_resource.download()
                except Exception as e:
                    logger.error(f"Failed to download HDX resource: {e}")
                    return None
                # hdx-python-api versions differ: (url, path) vs just path.
                if isinstance(dl_result, (list, tuple)) and len(dl_result) == 2:
                    _url, dl_path = dl_result
                else:
                    dl_path = dl_result
                if not dl_path:
                    logger.error("HDX resource download returned no path.")
                    return None
                dl_path = str(dl_path)

                if cache_path is not None:
                    import shutil as _shutil
                    try:
                        _shutil.move(dl_path, cache_path)
                        path_str = str(cache_path)
                    except Exception as mv_exc:
                        logger.debug(f"HDX cache move failed: {mv_exc}")
                        path_str = dl_path
                else:
                    path_str = dl_path

            # We should delete downloaded-but-not-cached files when done.
            _is_cached = (cache_path is not None and str(cache_path) == path_str)

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
                    if not _is_cached and os.path.exists(path_str):
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

                # Use pyogrio's bbox filter when available — avoids loading
                # the entire country file (Kontur PK is ~50 MB, >600k hexes).
                # The Kontur CRS is EPSG:3857 so reproject the bbox to match.
                try:
                    bbox_3857 = gpd.GeoSeries(
                        [box(minx, miny, maxx, maxy)], crs="EPSG:4326"
                    ).to_crs("EPSG:3857").total_bounds
                except Exception:
                    bbox_3857 = None

                try:
                    read_kwargs = {}
                    if bbox_3857 is not None:
                        read_kwargs["bbox"] = tuple(bbox_3857)
                    try:
                        pop_gdf_raw = gpd.read_file(gpkg_path, **read_kwargs)
                    except Exception:
                        # Some drivers/layers don't support bbox — retry without.
                        pop_gdf_raw = gpd.read_file(gpkg_path)
                except Exception as e:
                    logger.error(f"Failed to read HDX GeoPackage: {e}")
                    return None
                finally:
                    # Keep the cached original download; delete only the
                    # decompressed/renamed working copies and any non-cached
                    # direct download.
                    _paths_to_clean = set(extra_files)
                    if not _is_cached:
                        _paths_to_clean.add(path_str)
                    for p in _paths_to_clean:
                        if p and os.path.exists(p):
                            try:
                                os.unlink(p)
                            except Exception:
                                pass

                pop_gdf_raw = pop_gdf_raw.to_crs("EPSG:4326")

                # Secondary bbox clip (idempotent — safe even if pyogrio
                # already filtered).
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
                # Reproject to a metric CRS for accurate centroids, then back to WGS-84.
                # GeoPandas' centroid on WGS-84 emits a CRS warning and is
                # inaccurate near the poles, so we round-trip via EPSG:3857.
                centroids_3857 = gpd.GeoSeries(
                    clipped.to_crs("EPSG:3857").geometry.centroid,
                    crs="EPSG:3857",
                )
                clipped["geometry"] = centroids_3857.to_crs("EPSG:4326").values
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
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "en",
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
