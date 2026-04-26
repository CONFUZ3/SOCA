"""
NetworkManager — lazy-fetch and in-memory cache for OSMnx road-network graphs.

Lives in st.session_state["network_manager"] so the large graph objects are
never JSON-serialised into ADK session state.
"""

import hashlib
import logging
import threading
from contextlib import contextmanager
from typing import Optional, Tuple, Any

import geopandas as gpd

try:
    from utils.activity_log import timed as _timed
except Exception:  # pragma: no cover - activity_log is optional at import time
    _timed = None

try:
    from config.settings import settings as _soca_settings
except Exception:  # pragma: no cover
    _soca_settings = None

logger = logging.getLogger(__name__)


# Process-wide fetch locks + result cache, shared across NetworkManager
# instances. The prefetch thread and the solve-time fetch used to end up with
# separate NetworkManager objects (e.g. lazy creation in backend/api/aoi.py
# vs. chat.py), defeating per-instance locking and causing two 100+ second
# Overpass downloads for the same AOI. Keying coalesce state on the process
# — not the instance — closes that race.
_SHARED_FETCH_LOCKS: dict = {}
_SHARED_FETCH_RESULTS: dict = {}
_SHARED_REGISTRY_LOCK = threading.Lock()


class NetworkManager:
    CACHE_PRECISION = 2      # decimal places for bbox rounding (~1 km grid)
    MAX_CACHED_GRAPHS = 3
    DEFAULT_NETWORK_TYPE = "drive"
    BUFFER_DEGREES = 0.02    # ~2 km padding for bbox fallback
    # Simplify boundary polygons before hashing so that small float-drift
    # (e.g. unary_union on slightly-different inputs) doesn't produce
    # different cache keys for the same AOI.
    POLYGON_SIMPLIFY_TOL = 0.001  # ~100 m at the equator

    def __init__(self):
        self._cache: dict = {}
        self._cache_order: list = []
        # Retained for back-compat. Real coalescing now happens on the
        # module-level shared locks above.
        self._key_locks: dict = {}
        self._registry_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_graph(
        self,
        demand_gdf: gpd.GeoDataFrame,
        boundary_polygon=None,
    ) -> Tuple[Any, str]:
        """Return (G_proj, crs_proj). Fetches from OSM if not cached.

        Thread-safe: when two callers ask for the same AOI concurrently
        (typical case: background prefetch + solve-time fetch) the first
        caller performs the Overpass download while the second waits on a
        per-key lock and then reads the cached result.
        """
        import osmnx as ox

        cache_key = self._make_cache_key(demand_gdf, boundary_polygon)

        # Fast path — cache hit without holding any lock. Check both the
        # instance cache and the process-level shared cache.
        cached = self._cache_lookup(cache_key)
        if cached is not None:
            logger.info(
                "NetworkManager[%s]: cache hit for key %s",
                id(self), cache_key[:8],
            )
            return cached

        shared = _shared_result_lookup(cache_key)
        if shared is not None:
            # Promote into this instance's LRU so subsequent in-process hits
            # are fast.
            self._promote_to_instance_cache(cache_key, shared)
            logger.info(
                "NetworkManager[%s]: shared cache hit for key %s",
                id(self), cache_key[:8],
            )
            return shared

        # Serialise fetches per-key across the whole process. The first
        # caller fetches; any concurrent callers block here and then take
        # the cache-hit fast path.
        lock = _get_or_create_shared_lock(cache_key)
        with lock:
            cached = self._cache_lookup(cache_key) or _shared_result_lookup(cache_key)
            if cached is not None:
                self._promote_to_instance_cache(cache_key, cached)
                logger.info(
                    "NetworkManager[%s]: cache hit for key %s after lock (coalesced)",
                    id(self), cache_key[:8],
                )
                return cached

            logger.info(
                "NetworkManager[%s]: fetching road network from OSM (key=%s…)",
                id(self), cache_key[:8],
            )
            G = self._fetch_graph(ox, demand_gdf, boundary_polygon)
            G_proj, crs_proj = self._project_graph(ox, G)

            self._promote_to_instance_cache(cache_key, (G_proj, crs_proj))
            _shared_result_store(cache_key, (G_proj, crs_proj))
            logger.info("NetworkManager[%s]: graph cached (%d nodes)", id(self), len(G_proj))
            return G_proj, crs_proj

    # ------------------------------------------------------------------
    # Thread-safety helpers
    # ------------------------------------------------------------------

    def _cache_lookup(self, cache_key: str) -> Optional[Tuple[Any, str]]:
        """Thread-safe cache get + LRU touch."""
        with self._registry_lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                return None
            try:
                self._cache_order.remove(cache_key)
            except ValueError:
                pass
            self._cache_order.append(cache_key)
            return entry["G_proj"], entry["crs_proj"]

    def _get_or_create_key_lock(self, cache_key: str) -> threading.Lock:
        """Back-compat: return the module-level shared lock for *cache_key*."""
        return _get_or_create_shared_lock(cache_key)

    def _promote_to_instance_cache(self, cache_key: str, result):
        """Store a (G_proj, crs_proj) result in this instance's LRU cache."""
        G_proj, crs_proj = result
        with self._registry_lock:
            if cache_key in self._cache:
                try:
                    self._cache_order.remove(cache_key)
                except ValueError:
                    pass
            self._cache[cache_key] = {"G_proj": G_proj, "crs_proj": crs_proj}
            self._cache_order.append(cache_key)
            self._evict_if_needed_locked()

    def is_osmnx_available(self) -> bool:
        try:
            import osmnx  # noqa: F401
            return True
        except ImportError:
            return False

    def clear_cache(self):
        with self._registry_lock:
            self._cache.clear()
            self._cache_order.clear()
            self._key_locks.clear()
        # Also clear the process-wide shared caches, otherwise the next
        # get_graph() for a previously-seen key would hit the shared result
        # cache and skip the fetch — defeating the point of clear_cache().
        with _SHARED_REGISTRY_LOCK:
            _SHARED_FETCH_RESULTS.clear()
            _SHARED_FETCH_LOCKS.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_cache_key(self, demand_gdf: gpd.GeoDataFrame, boundary_polygon) -> str:
        if boundary_polygon is not None:
            # Simplify to absorb float drift between slightly-different
            # polygon representations (e.g. unary_union on inputs with
            # different geometry ordering). Falls back to raw WKT if
            # simplify raises on a degenerate geometry.
            try:
                simplified = boundary_polygon.simplify(
                    self.POLYGON_SIMPLIFY_TOL, preserve_topology=True
                )
                raw = simplified.wkt if not simplified.is_empty else boundary_polygon.wkt
            except Exception:
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
        ox.settings.requests_timeout = int(
            getattr(_soca_settings, "OSMNX_REQUESTS_TIMEOUT", 60)
        ) if _soca_settings is not None else 60
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
                area_km2 = self._polygon_area_km2(boundary_polygon)
                detail = f"polygon, AOI={area_km2:.0f} km^2"
                with self._activity("network.fetch", detail) as evt:
                    G = ox.graph_from_polygon(boundary_polygon, network_type=network_type)
                    if evt is not None:
                        evt.detail = f"{detail}, graph: {len(G.nodes)} nodes, {len(G.edges)} edges"
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
        detail = f"bbox W={west:.3f} S={south:.3f} E={east:.3f} N={north:.3f}"
        with self._activity("network.fetch", detail) as evt:
            G = ox.graph_from_bbox(
                bbox=(west, south, east, north),
                network_type=network_type,
            )
            if evt is not None:
                evt.detail = f"{detail}, graph: {len(G.nodes)} nodes, {len(G.edges)} edges"
        logger.info(
            "NetworkManager: fetched graph from bbox (W=%.4f S=%.4f E=%.4f N=%.4f)",
            west, south, east, north,
        )
        return G

    # ------------------------------------------------------------------
    # Activity-log helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _activity(self, stage: str, detail: str):
        """Yield the timed event (or None if activity_log unavailable)."""
        if _timed is None:
            yield None
            return
        with _timed(stage, source="OpenStreetMap", detail=detail) as evt:
            yield evt

    @staticmethod
    def _polygon_area_km2(polygon) -> float:
        """Return the polygon area in km^2 using a WGS84 geodesic computation."""
        try:
            from pyproj import Geod
            geod = Geod(ellps="WGS84")
            area_m2, _ = geod.geometry_area_perimeter(polygon)
            return abs(area_m2) / 1_000_000.0
        except Exception:
            return 0.0

    def _project_graph(self, ox, G) -> Tuple[Any, str]:
        G_proj = ox.project_graph(G)
        crs_proj = str(G_proj.graph.get("crs", "EPSG:3857"))
        return G_proj, crs_proj

    def _evict_if_needed(self):
        with self._registry_lock:
            self._evict_if_needed_locked()

    def _evict_if_needed_locked(self):
        while len(self._cache) > self.MAX_CACHED_GRAPHS and self._cache_order:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
            logger.debug("NetworkManager: evicted cache entry %s…", oldest[:8])


# ----------------------------------------------------------------------
# Module-level shared fetch coalescing
# ----------------------------------------------------------------------


def _get_or_create_shared_lock(cache_key: str) -> threading.Lock:
    with _SHARED_REGISTRY_LOCK:
        lock = _SHARED_FETCH_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _SHARED_FETCH_LOCKS[cache_key] = lock
        return lock


def _shared_result_lookup(cache_key: str):
    with _SHARED_REGISTRY_LOCK:
        return _SHARED_FETCH_RESULTS.get(cache_key)


def _shared_result_store(cache_key: str, result) -> None:
    with _SHARED_REGISTRY_LOCK:
        _SHARED_FETCH_RESULTS[cache_key] = result
        # Cap the shared cache at 8 entries (process-wide). Much larger than
        # the per-session MAX_CACHED_GRAPHS to absorb a few concurrent AOIs.
        if len(_SHARED_FETCH_RESULTS) > 8:
            # Drop an arbitrary oldest-ish entry; we don't need strict LRU
            # here — this exists to deduplicate fetches, not serve as a
            # long-lived cache.
            oldest = next(iter(_SHARED_FETCH_RESULTS))
            _SHARED_FETCH_RESULTS.pop(oldest, None)


# ----------------------------------------------------------------------
# Background pre-fetch helpers (used by app.py on AOI confirmation)
# ----------------------------------------------------------------------

NETWORK_STATUS_KEY = "_network_status"
NETWORK_STATUS_ERROR_KEY = "_network_status_error"
NETWORK_STATUS_STATS_KEY = "_network_status_stats"


def prefetch_network_graph(
    network_manager: "NetworkManager",
    aoi_gdf: gpd.GeoDataFrame,
    session_state=None,
) -> None:
    """Fetch the road graph for *aoi_gdf* and write status to *session_state*.

    Intended to be called as the target of a daemon thread launched from
    `app.py` right after the user confirms an AOI. Safe to call on the main
    thread too (used by unit tests). Never raises — errors are captured and
    written as "failed" status.

    session_state: any dict-like (st.session_state or a plain dict for tests).
    """
    def _set(key: str, value: Any) -> None:
        if session_state is None:
            return
        try:
            session_state[key] = value
        except Exception:
            logger.debug("prefetch_network_graph: session_state write for %s failed", key)

    _set(NETWORK_STATUS_KEY, "fetching")
    _set(NETWORK_STATUS_ERROR_KEY, None)

    try:
        if aoi_gdf is None or len(aoi_gdf) == 0:
            raise ValueError("AOI GeoDataFrame is empty")
        if not network_manager.is_osmnx_available():
            raise RuntimeError("osmnx is not installed")

        boundary_polygon = None
        try:
            from shapely.ops import unary_union
            boundary_polygon = unary_union(aoi_gdf.geometry)
        except Exception:
            boundary_polygon = aoi_gdf.geometry.iloc[0]

        G_proj, _crs = network_manager.get_graph(aoi_gdf, boundary_polygon=boundary_polygon)
        n_nodes = len(G_proj.nodes) if hasattr(G_proj, "nodes") else 0
        n_edges = len(G_proj.edges) if hasattr(G_proj, "edges") else 0
        _set(NETWORK_STATUS_STATS_KEY, {"nodes": n_nodes, "edges": n_edges})
        _set(NETWORK_STATUS_KEY, "ready")
    except Exception as exc:
        logger.warning("prefetch_network_graph: failed (%s)", exc)
        _set(NETWORK_STATUS_ERROR_KEY, str(exc))
        _set(NETWORK_STATUS_KEY, "failed")


def launch_prefetch_thread(
    network_manager: "NetworkManager",
    aoi_gdf: gpd.GeoDataFrame,
    session_state=None,
):
    """Launch `prefetch_network_graph` in a daemon thread, returning the Thread.

    When Streamlit is available, the parent script's run context is attached
    to the thread so that `st.session_state` and the activity log are safely
    accessible from the worker.
    """
    import threading

    thread = threading.Thread(
        target=prefetch_network_graph,
        args=(network_manager, aoi_gdf, session_state),
        name="soca-network-prefetch",
        daemon=True,
    )

    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx  # type: ignore
        add_script_run_ctx(thread)
    except Exception:
        pass  # not running under streamlit (e.g. tests) — no context to attach

    thread.start()
    return thread
