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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DataFetchError(Exception):
    """Base exception for all data-fetching failures."""


class GeocodingError(DataFetchError):
    """Raised when Nominatim geocoding / boundary retrieval fails."""


class OverpassError(DataFetchError):
    """Raised when the Overpass API returns an error or empty result."""


class PopulationDataError(DataFetchError):
    """Raised when synthetic population-grid generation fails."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOMINATIM_URL = "https://nominatim.openstreetmap.org"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HDX_BASE_URL = "https://data.humdata.org/api/3/action"
PHOTON_URL = "https://photon.komoot.io"
# Alternative Overpass instances (used as final fallback)
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Overture Maps Place Category Mapping (Singular forms as per 2024/2025 taxonomy)
OVERTURE_CATEGORIES: dict[str, list[str]] = {
    "health": ["hospital", "medical_clinic", "doctor", "pharmacy", "medical_center", "health_center"],
    "education": ["school", "university", "college", "kindergarten", "preschool"],
    "food": ["supermarket", "grocery_store", "convenience_store", "market"],
    "finance": ["bank", "atm"],
    "fire_station": ["fire_station"],
    "police": ["police_station"],
    "library": ["library"],
}

# Legacy Overpass QL tag filters (kept for fallback)
FACILITY_TAGS: dict[str, list[str]] = {
    "health": [
        '["amenity"~"hospital|clinic|health_centre|doctors|pharmacy"]',
    ],
    "education": [
        '["amenity"~"school|university|college|kindergarten"]',
    ],
    "food": [
        '["amenity"~"marketplace|supermarket"]',
        '["shop"~"supermarket|convenience|grocery"]',
    ],
    "finance": [
        '["amenity"~"bank|atm"]',
    ],
    "fire_station": [
        '["amenity"="fire_station"]',
    ],
    "police": [
        '["amenity"="police"]',
    ],
    "library": [
        '["amenity"="library"]',
    ],
}

# Default total synthetic population spread over all grid points
_DEFAULT_TOTAL_POPULATION = 100_000

# Retry parameters
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1  # seconds — doubles each retry


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

    def fetch_boundaries(self, location: str) -> gpd.GeoDataFrame:
        """
        Fetch the administrative boundary polygon for *location*.

        Uses a 3-tier fallback chain to maximise reliability:
          1. Overpass API admin-boundary relation query (same OSM data,
             different endpoint — avoids Nominatim 403 blocks).
          2. Photon geocoder (komoot.io) with bbox polygon construction.
          3. Nominatim as a last resort.

        Args:
            location: Human-readable place name, e.g. ``"Lima, Peru"``.

        Returns:
            GeoDataFrame with one row containing the boundary polygon.
            CRS is WGS-84 (EPSG:4326).

        Raises:
            GeocodingError: If all backends fail or return no polygon.
        """
        logger.info(f"DataFetcher: Fetching boundary for '{location}'")

        # --- Tier 0: Overture Maps division_area (fastest, actual polygon) -
        if _OVERTURE_AVAILABLE:
            try:
                gdf = self._fetch_boundary_via_overture(location)
                return gdf
            except Exception as exc:
                logger.warning(
                    f"Overture boundary fetch failed for '{location}': {exc}. "
                    "Falling back to Photon."
                )

        # --- Tier 1: Photon geocoder (komoot.io) --------------------------
        try:
            gdf = self._fetch_boundary_via_photon(location)
            logger.info(
                f"DataFetcher: Boundary for '{location}' obtained via Photon "
                f"(geom type: {gdf.geometry.iloc[0].geom_type})"
            )
            return gdf
        except Exception as exc:
            logger.warning(
                f"Photon boundary fetch failed for '{location}': {exc}. "
                "Falling back to Overpass."
            )

        # --- Tier 2: Overpass relation boundary search --------------------
        try:
            gdf = self._fetch_boundary_via_overpass(location)
            logger.info(
                f"DataFetcher: Boundary for '{location}' obtained via Overpass "
                f"(geom type: {gdf.geometry.iloc[0].geom_type})"
            )
            return gdf
        except Exception as exc:
            logger.warning(
                f"Overpass boundary fetch failed for '{location}': {exc}. "
                "Falling back to Nominatim."
            )

        # --- Tier 3: Nominatim (last resort) ------------------------------
        try:
            gdf = self._fetch_boundary_via_nominatim(location)
            logger.info(
                f"DataFetcher: Boundary for '{location}' obtained via Nominatim "
                f"(geom type: {gdf.geometry.iloc[0].geom_type})"
            )
            return gdf
        except Exception as exc:
            raise GeocodingError(
                f"All geocoding backends failed for '{location}'. "
                f"Last error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Boundary backend implementations
    # ------------------------------------------------------------------

    def _fetch_boundary_via_overpass(self, location: str) -> gpd.GeoDataFrame:
        """
        Fetch boundary polygon via Overpass ``relation[boundary=administrative]``.

        This queries the same OSM dataset as Nominatim but through a completely
        independent endpoint, sidestepping Nominatim rate-limiting / IP blocks.
        """
        # Search multiple admin_level values (4=state/province, 5, 6=district, 8=city)
        # The query uses Overpass's ``name:en`` + ``name`` matching.
        escaped = location.replace('"', r'\"')
        query = (
            "[out:json][timeout:30];\n"
            "(\n"
            f'  relation["boundary"="administrative"]["name"~"{escaped}",i];\n'
            f'  relation["boundary"="administrative"]["name:en"~"{escaped}",i];\n'
            ");\n"
            "out geom;"
        )

        # Try each Overpass mirror
        last_exc: Optional[Exception] = None
        for mirror_url in OVERPASS_MIRRORS:
            try:
                resp = self._make_request(
                    mirror_url, params={"data": query}, method="POST", timeout=60
                )
                data = resp.json()
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(f"Overpass mirror {mirror_url} failed: {exc}")
        else:
            raise DataFetchError(
                f"All Overpass mirrors failed for boundary query. Last: {last_exc}"
            )

        elements = data.get("elements", [])
        if not elements:
            raise GeocodingError(
                f"Overpass returned no admin-boundary relations for '{location}'."
            )

        # Pick the relation with the highest admin_level (most specific)
        def _admin_level_sort(el: dict) -> int:
            try:
                return int(el.get("tags", {}).get("admin_level", 0))
            except (ValueError, TypeError):
                return 0

        elements_sorted = sorted(elements, key=_admin_level_sort, reverse=True)

        # Reconstruct polygon from outer members
        geom = self._overpass_relation_to_shape(elements_sorted[0])
        if geom is None or geom.is_empty:
            raise GeocodingError(
                f"Could not reconstruct polygon from Overpass relation for '{location}'."
            )

        tags = elements_sorted[0].get("tags", {})
        props = {
            "name": tags.get("name:en", tags.get("name", location)),
            "location_query": location,
            "source": "overpass_boundary",
            "admin_level": tags.get("admin_level", ""),
        }
        return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")

    def _overpass_relation_to_shape(self, element: dict):
        """Convert an Overpass relation element (with ``geometry`` members) to a Shapely geometry."""
        from shapely.geometry import LinearRing, Polygon, MultiPolygon
        from shapely.ops import polygonize, unary_union

        outer_coords: list[list] = []
        inner_coords: list[list] = []

        members = element.get("members", [])
        for member in members:
            if member.get("type") != "way":
                continue
            role = member.get("role", "outer")
            geometry = member.get("geometry", [])
            coords = [(pt["lon"], pt["lat"]) for pt in geometry if "lon" in pt and "lat" in pt]
            if len(coords) < 2:
                continue
            if role == "outer":
                outer_coords.append(coords)
            elif role == "inner":
                inner_coords.append(coords)

        if not outer_coords:
            # Fallback: some relations store geometry directly as nodes
            geometry = element.get("geometry", [])
            if geometry:
                coords = [(pt["lon"], pt["lat"]) for pt in geometry if "lon" in pt and "lat" in pt]
                if len(coords) >= 3:
                    return Polygon(coords)
            return None

        # Build outer polygon(s) via polygonize
        from shapely.geometry import LineString
        outer_lines = [LineString(c) for c in outer_coords if len(c) >= 2]
        outer_polys = list(polygonize(outer_lines))

        if not outer_polys:
            # Try directly if ways are already closed rings
            outer_polys = []
            for c in outer_coords:
                if len(c) >= 3:
                    try:
                        poly = Polygon(c)
                        if poly.is_valid:
                            outer_polys.append(poly)
                    except Exception:
                        pass

        if not outer_polys:
            return None

        outer_geom = unary_union(outer_polys)

        # Subtract inner (holes)
        if inner_coords:
            inner_lines = [LineString(c) for c in inner_coords if len(c) >= 2]
            inner_polys = list(polygonize(inner_lines))
            if inner_polys:
                inner_geom = unary_union(inner_polys)
                outer_geom = outer_geom.difference(inner_geom)

        return outer_geom

    def _fetch_boundary_via_photon(self, location: str) -> gpd.GeoDataFrame:
        """
        Fetch boundary via Photon geocoder (photon.komoot.io).

        When Photon returns an OSM relation ID we try to retrieve the actual
        polygon from Overpass (same data, just via Photon for the lookup).
        If that fails we fall back to the extent/bbox bounding box so callers
        always get a polygon.  The bbox fallback is intentionally the *last*
        resort because a plain rectangle often covers the ocean.
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

        # Prefer features that have an ``extent`` (bbox of the actual polygon)
        best = None
        for feat in features:
            p = feat.get("properties", {})
            # Prefer administrative-type results with an OSM relation
            if p.get("osm_type") == "R" and p.get("extent"):
                best = feat
                break
            if p.get("extent"):
                best = feat  # take first with extent, keep looking for relation
        if best is None:
            best = features[0]

        props_raw = best.get("properties", {})

        # --- Tier A: Try to get a real polygon via Overpass using the OSM relation ID
        osm_type = props_raw.get("osm_type", "")
        osm_id   = props_raw.get("osm_id")
        if osm_type == "R" and osm_id:
            try:
                rel_query = (
                    "[out:json][timeout:30];\n"
                    f"relation({osm_id});\n"
                    "out geom;"
                )
                for mirror_url in OVERPASS_MIRRORS:
                    try:
                        rel_resp = self._make_request(
                            mirror_url, params={"data": rel_query},
                            method="POST", timeout=60
                        )
                        rel_data = rel_resp.json()
                        break
                    except Exception:
                        continue
                else:
                    rel_data = {}

                rel_elements = rel_data.get("elements", [])
                if rel_elements:
                    geom = self._overpass_relation_to_shape(rel_elements[0])
                    if geom is not None and not geom.is_empty:
                        tags = rel_elements[0].get("tags", {})
                        props = {
                            "name": tags.get("name:en", tags.get("name", props_raw.get("name", location))),
                            "location_query": location,
                            "source": "photon_then_overpass",
                            "country": props_raw.get("country", ""),
                        }
                        logger.info(
                            f"DataFetcher: Retrieved actual polygon from Overpass "
                            f"for OSM relation {osm_id} (from Photon lookup)"
                        )
                        return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")
            except Exception as rel_exc:
                logger.warning(
                    f"Could not fetch Overpass polygon for OSM relation {osm_id}: {rel_exc}"
                )

        # --- Tier B: Use the extent bbox (rectangular — less accurate)
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
        return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")

    def _fetch_boundary_via_overture(self, location: str) -> gpd.GeoDataFrame:
        """Fetch administrative boundary polygon from Overture Maps division_area theme.

        Uses Photon for a quick lat/lon geocode to seed the bbox, then queries
        Overture's division_area theme for the actual polygon. Returns the
        largest name-matched administrative area (locality, county, region, or country).
        """
        if not _OVERTURE_AVAILABLE:
            raise DataFetchError("overturemaps package not available")

        import shapely.wkb as wkb

        # Step 1: Get a rough lat/lon from Photon to seed the bbox
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
        buf = 2.0
        bbox = (lon - buf, lat - buf, lon + buf, lat + buf)

        logger.info(f"DataFetcher: Querying Overture division_area for '{location}' bbox={bbox}")

        # Step 2: Query Overture division_area theme
        try:
            reader = overturemaps.record_batch_reader("division_area", bbox=bbox)
            if reader is None:
                raise DataFetchError("Overture reader returned None")
            table = reader.read_all()
        except Exception as exc:
            raise DataFetchError(f"Overture division_area query failed: {exc}") from exc

        if table.num_rows == 0:
            raise DataFetchError(f"Overture returned no division_area results near '{location}'")

        # Step 3: Filter to admin subtypes and match on name
        df = table.to_pandas()

        ADMIN_SUBTYPES = {"locality", "county", "region", "country"}
        if "subtype" in df.columns:
            df = df[df["subtype"].isin(ADMIN_SUBTYPES)]

        if df.empty:
            raise DataFetchError(f"No admin-level division_area found near '{location}'")

        def _primary_name(names_val):
            if isinstance(names_val, dict):
                return names_val.get("primary") or ""
            return ""

        df = df.copy()
        df["_name"] = df["names"].apply(_primary_name)

        # Match on the first part of the query (before any comma)
        primary_query = location.split(",")[0].strip().lower()
        name_match = df[df["_name"].str.lower().str.contains(primary_query, na=False, regex=False)]

        if name_match.empty:
            raise DataFetchError(f"No Overture division_area matched name '{location}'")

        # Pick the largest polygon among matches
        geoms = [wkb.loads(g) for g in name_match["geometry"]]
        areas = [g.area for g in geoms]
        best_idx = areas.index(max(areas))
        geom = geoms[best_idx]
        best_row = name_match.iloc[best_idx]

        props = {
            "name": best_row["_name"],
            "location_query": location,
            "source": "overture_division_area",
            "subtype": best_row.get("subtype", ""),
        }
        logger.info(
            f"DataFetcher: Boundary for '{location}' obtained via Overture "
            f"(subtype={props['subtype']}, name='{props['name']}')"
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
        for speed and reliability. Falls back to Overpass if Overture
        is unavailable or fails.

        Args:
            boundary_gdf: GeoDataFrame with at least one geometry row
                          representing the region of interest.
            category: One of the keys in :data:`OVERTURE_CATEGORIES` or :data:`FACILITY_TAGS`.

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
                logger.warning(f"Overture POI fetch failed for '{category}': {exc}. Falling back to Overpass.")

        # --- Tier 2: Overpass (Legacy Fallback) ---------------------------
        try:
            pois_gdf = self._fetch_pois_via_overpass(boundary_gdf, category)
            return pois_gdf
        except Exception as exc:
            raise OverpassError(
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

    def _fetch_pois_via_overpass(
        self,
        boundary_gdf: gpd.GeoDataFrame,
        category: str,
    ) -> gpd.GeoDataFrame:
        """Original Overpass POI fetcher refactored as a helper."""
        if category not in FACILITY_TAGS:
            raise OverpassError(
                f"Unknown POI category '{category}' for Overpass. "
                f"Supported: {sorted(FACILITY_TAGS.keys())}"
            )

        tag_filters = FACILITY_TAGS[category]

        # Get bounding box in (south, west, north, east) Overpass convention
        bounds = boundary_gdf.to_crs("EPSG:4326").total_bounds  # minx, miny, maxx, maxy
        bbox_str = f"{bounds[1]},{bounds[0]},{bounds[3]},{bounds[2]}"

        # Build compound Overpass QL query (node + way for each tag filter, OR'd)
        union_parts: list[str] = []
        for tag_filter in tag_filters:
            union_parts.append(f"node{tag_filter}({bbox_str});")
            union_parts.append(f"way{tag_filter}({bbox_str});")

        query = (
            "[out:json][timeout:60];\n"
            "(\n"
            + "".join(f"  {part}\n" for part in union_parts)
            + ");\n"
            "out center;"
        )

        logger.info(
            f"DataFetcher: Querying Overpass (Fallback) for category='{category}' "
            f"bbox={bbox_str}"
        )

        # Try each Overpass mirror in sequence
        last_exc: Optional[Exception] = None
        response = None
        for mirror_url in OVERPASS_MIRRORS:
            try:
                response = self._make_request(
                    mirror_url,
                    params={"data": query},
                    method="POST",
                    timeout=120,
                )
                break  # success
            except DataFetchError as exc:
                last_exc = exc
                logger.warning(
                    f"Overpass POI mirror {mirror_url} failed: {exc}. Trying next..."
                )

        if response is None:
            raise OverpassError(
                f"Network error fetching POIs ({category}): All Overpass mirrors failed. "
                f"Last error: {last_exc}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise OverpassError(
                f"Invalid JSON from Overpass for category '{category}': {exc}"
            ) from exc

        elements = data.get("elements", [])
        if not elements:
            return gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")

        rows = []
        for el in elements:
            if el.get("type") == "node":
                lat, lon = el.get("lat"), el.get("lon")
            else:
                center = el.get("center", {})
                lat, lon = center.get("lat"), center.get("lon")

            if lat is None or lon is None:
                continue

            tags = el.get("tags", {})
            rows.append({
                "name": tags.get("name", ""),
                "amenity": tags.get("amenity", tags.get("shop", "")),
                "geometry": Point(lon, lat),
            })

        if not rows:
            return gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")

        pois_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        return self._clip_to_boundary(pois_gdf, boundary_gdf)

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
        n_points: int = 200,
        random_seed: int = 42,
    ) -> gpd.GeoDataFrame:
        """
        Generate a synthetic population grid within *boundary_gdf*.

        Uses rejection sampling to place *n_points* random Point geometries
        strictly inside the boundary polygon (including water-exclusion when
        the boundary geometry is a proper polygon rather than a rectangle).
        Each point is assigned a population weight so the **total** always
        equals :data:`_DEFAULT_TOTAL_POPULATION`.

        Prior to generating synthetic data, this method attempts to fetch
        real population estimates from the Humanitarian Data Exchange (HDX)
        Facebook high-resolution maps. The synthetic fallback is used when
        real-data sources fail or are too large to process in memory safely.

        Args:
            boundary_gdf: Region of interest (any CRS; reprojected internally).
            n_points: Number of sample points to generate.
            random_seed: Seed for the internal RNG (default 42 for
                backward-compatibility).  Pass different values to get
                statistically independent realisations.

        Returns:
            GeoDataFrame with ``population`` column and Point geometry.
            CRS is EPSG:4326.

        Raises:
            PopulationDataError: If the boundary polygon is invalid or empty.
        """
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

        # Recompute pop_per_point from actual placed count so the total
        # population always equals _DEFAULT_TOTAL_POPULATION regardless of
        # how many points fit inside the (possibly thin) boundary polygon.
        pop_per_point = _DEFAULT_TOTAL_POPULATION / len(points)

        gdf = gpd.GeoDataFrame(
            {"population": [pop_per_point] * len(points)},
            geometry=points,
            crs="EPSG:4326",
        )
        gdf["data_source"] = "synthetic_uniform_grid"
        logger.info(
            f"DataFetcher: Generated {len(gdf)} synthetic population points "
            f"({pop_per_point:.1f} pop each, total={_DEFAULT_TOTAL_POPULATION})"
        )
        return gdf

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
                        country_raw = features[0].get("properties", {}).get("country", "")
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
            all_candidates: list[tuple] = []
            for query in [
                f"title:{country_norm}-high-resolution-population-density-maps-demographic-estimates",
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
            try:
                url, path = target_resource.download()
            except Exception as e:
                logger.error(f"Failed to download HDX resource: {e}")
                return None

            if not path:
                logger.error("HDX resource download returned no path.")
                return None

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
