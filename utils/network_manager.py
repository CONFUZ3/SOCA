"""
NetworkManager — lazy-fetch and in-memory cache for OSMnx road-network graphs.

Lives in st.session_state["network_manager"] so the large graph objects are
never JSON-serialised into ADK session state.
"""

import hashlib
import logging
from typing import Optional, Tuple, Any

import geopandas as gpd

logger = logging.getLogger(__name__)


class NetworkManager:
    CACHE_PRECISION = 2      # decimal places for bbox rounding (~1 km grid)
    MAX_CACHED_GRAPHS = 3
    DEFAULT_NETWORK_TYPE = "drive"
    BUFFER_DEGREES = 0.02    # ~2 km padding for bbox fallback

    def __init__(self):
        self._cache: dict = {}
        self._cache_order: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_graph(
        self,
        demand_gdf: gpd.GeoDataFrame,
        boundary_polygon=None,
    ) -> Tuple[Any, str]:
        """Return (G_proj, crs_proj). Fetches from OSM if not cached."""
        import osmnx as ox

        cache_key = self._make_cache_key(demand_gdf, boundary_polygon)
        if cache_key in self._cache:
            logger.info("NetworkManager: cache hit for key %s", cache_key[:8])
            entry = self._cache[cache_key]
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return entry["G_proj"], entry["crs_proj"]

        logger.info("NetworkManager: fetching road network from OSM (key=%s…)", cache_key[:8])
        G = self._fetch_graph(ox, demand_gdf, boundary_polygon)
        G_proj, crs_proj = self._project_graph(ox, G)

        self._evict_if_needed()
        self._cache[cache_key] = {"G_proj": G_proj, "crs_proj": crs_proj}
        self._cache_order.append(cache_key)
        logger.info("NetworkManager: graph cached (%d nodes)", len(G_proj))
        return G_proj, crs_proj

    def is_osmnx_available(self) -> bool:
        try:
            import osmnx  # noqa: F401
            return True
        except ImportError:
            return False

    def clear_cache(self):
        self._cache.clear()
        self._cache_order.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_cache_key(self, demand_gdf: gpd.GeoDataFrame, boundary_polygon) -> str:
        if boundary_polygon is not None:
            raw = boundary_polygon.wkt
        else:
            # Round bbox to CACHE_PRECISION to create a coarse grid key
            bounds = demand_gdf.total_bounds  # (minx, miny, maxx, maxy)
            p = self.CACHE_PRECISION
            raw = f"{round(bounds[0], p)},{round(bounds[1], p)},{round(bounds[2], p)},{round(bounds[3], p)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _fetch_graph(self, ox, demand_gdf: gpd.GeoDataFrame, boundary_polygon) -> Any:
        # Shared osmnx settings: enable disk cache + identify the app so we
        # comply with Nominatim/Overpass ToS (generic library UA is rejected).
        ox.settings.use_cache = True
        ox.settings.requests_timeout = 180
        ox.settings.overpass_rate_limit = True
        ox.settings.log_console = False
        # Nominatim's ToS blocks placeholder-domain UA strings. Identify via
        # the public repo URL so this client isn't rate-limited / blocked.
        ox.settings.http_user_agent = (
            "SOCA-spopt/1.0 (Spatial Optimization Conversational Agent; "
            "academic research; +https://github.com/soca-spopt/soca)"
        )

        network_type = self.DEFAULT_NETWORK_TYPE
        if boundary_polygon is not None:
            try:
                G = ox.graph_from_polygon(boundary_polygon, network_type=network_type)
                logger.info("NetworkManager: fetched graph from boundary polygon")
                return G
            except Exception as exc:
                logger.warning("NetworkManager: polygon fetch failed (%s), falling back to bbox", exc)

        # Bbox fallback with padding. osmnx 2.x expects bbox=(left, bottom,
        # right, top) as a keyword argument — i.e. (west, south, east, north).
        bounds = demand_gdf.to_crs("EPSG:4326").total_bounds  # minx, miny, maxx, maxy
        buf = self.BUFFER_DEGREES
        west, south = bounds[0] - buf, bounds[1] - buf
        east, north = bounds[2] + buf, bounds[3] + buf
        G = ox.graph_from_bbox(
            bbox=(west, south, east, north),
            network_type=network_type,
        )
        logger.info("NetworkManager: fetched graph from bbox (W=%.4f S=%.4f E=%.4f N=%.4f)", west, south, east, north)
        return G

    def _project_graph(self, ox, G) -> Tuple[Any, str]:
        G_proj = ox.project_graph(G)
        crs_proj = str(G_proj.graph.get("crs", "EPSG:3857"))
        return G_proj, crs_proj

    def _evict_if_needed(self):
        while len(self._cache) >= self.MAX_CACHED_GRAPHS and self._cache_order:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
            logger.debug("NetworkManager: evicted cache entry %s…", oldest[:8])
