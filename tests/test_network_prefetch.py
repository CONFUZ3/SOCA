"""Unit tests for the road-network prefetch helpers in utils.network_manager."""

import sys
import threading
import time
from unittest.mock import MagicMock, patch

import geopandas as gpd
from shapely.geometry import Polygon

from utils import network_manager as nm_mod
from utils.network_manager import NetworkManager


def _make_aoi_gdf():
    poly = Polygon([(74.30, 31.50), (74.40, 31.50), (74.40, 31.60), (74.30, 31.60)])
    return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")


def test_prefetch_sets_ready_status_on_success():
    G = MagicMock()
    G.nodes = range(100)
    G.edges = range(250)

    nm = MagicMock()
    nm.is_osmnx_available.return_value = True
    nm.get_graph.return_value = (G, "EPSG:32642")

    state: dict = {}
    nm_mod.prefetch_network_graph(nm, _make_aoi_gdf(), session_state=state)

    assert state[nm_mod.NETWORK_STATUS_KEY] == "ready"
    assert state[nm_mod.NETWORK_STATUS_STATS_KEY] == {"nodes": 100, "edges": 250}
    assert state[nm_mod.NETWORK_STATUS_ERROR_KEY] is None
    nm.get_graph.assert_called_once()


def test_prefetch_sets_failed_status_on_fetch_error():
    nm = MagicMock()
    nm.is_osmnx_available.return_value = True
    nm.get_graph.side_effect = RuntimeError("overpass down")

    state: dict = {}
    nm_mod.prefetch_network_graph(nm, _make_aoi_gdf(), session_state=state)

    assert state[nm_mod.NETWORK_STATUS_KEY] == "failed"
    assert "overpass down" in state[nm_mod.NETWORK_STATUS_ERROR_KEY]


def test_prefetch_sets_failed_status_when_osmnx_missing():
    nm = MagicMock()
    nm.is_osmnx_available.return_value = False

    state: dict = {}
    nm_mod.prefetch_network_graph(nm, _make_aoi_gdf(), session_state=state)

    assert state[nm_mod.NETWORK_STATUS_KEY] == "failed"
    assert "osmnx" in state[nm_mod.NETWORK_STATUS_ERROR_KEY]
    nm.get_graph.assert_not_called()


def test_prefetch_handles_empty_aoi():
    nm = MagicMock()
    nm.is_osmnx_available.return_value = True

    state: dict = {}
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    nm_mod.prefetch_network_graph(nm, empty, session_state=state)

    assert state[nm_mod.NETWORK_STATUS_KEY] == "failed"
    assert "empty" in state[nm_mod.NETWORK_STATUS_ERROR_KEY].lower()


def test_prefetch_no_session_state_is_noop_safe():
    """Passing session_state=None must not raise."""
    G = MagicMock()
    G.nodes = range(3)
    G.edges = range(2)

    nm = MagicMock()
    nm.is_osmnx_available.return_value = True
    nm.get_graph.return_value = (G, "EPSG:32642")

    nm_mod.prefetch_network_graph(nm, _make_aoi_gdf(), session_state=None)
    nm.get_graph.assert_called_once()


def test_launch_prefetch_thread_runs_target():
    G = MagicMock()
    G.nodes = range(5)
    G.edges = range(4)
    nm = MagicMock()
    nm.is_osmnx_available.return_value = True
    nm.get_graph.return_value = (G, "EPSG:32642")

    state: dict = {}
    thread = nm_mod.launch_prefetch_thread(nm, _make_aoi_gdf(), session_state=state)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert state[nm_mod.NETWORK_STATUS_KEY] == "ready"


# ---------------------------------------------------------------------------
# Concurrency lock: two threads asking for the same graph must coalesce into
# a single Overpass fetch. This is the guard that prevents the prefetch +
# confirm_optimization race the terminal log exposed.
# ---------------------------------------------------------------------------

class _FakeSettings:
    """Stand-in for osmnx.settings with permissive attribute assignment."""

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


class _FakeOSMnx:
    """Minimal osmnx stand-in for graph construction."""

    def __init__(self, fetch_delay: float = 0.2):
        self.fetch_delay = fetch_delay
        self.fetch_count = 0
        self._lock = threading.Lock()
        self.settings = _FakeSettings()

    def graph_from_polygon(self, polygon, network_type=None):
        with self._lock:
            self.fetch_count += 1
        time.sleep(self.fetch_delay)
        G = MagicMock()
        G.nodes = list(range(10))
        G.edges = list(range(15))
        G.graph = {"crs": "EPSG:4326"}
        return G

    def graph_from_bbox(self, bbox=None, network_type=None):
        return self.graph_from_polygon(None)

    def project_graph(self, G):
        G.graph = {"crs": "EPSG:32642"}
        return G


def test_get_graph_coalesces_concurrent_fetches(monkeypatch):
    """Two threads fetching the same AOI should trigger exactly ONE osmnx call."""
    fake_ox = _FakeOSMnx(fetch_delay=0.25)

    # Inject the fake osmnx as `import osmnx` inside get_graph.
    monkeypatch.setitem(sys.modules, "osmnx", fake_ox)

    nm = NetworkManager()
    aoi = _make_aoi_gdf()
    polygon = aoi.geometry.iloc[0]

    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(nm.get_graph(aoi, boundary_polygon=polygon))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"worker raised: {errors!r}"
    assert len(results) == 4
    # All callers received the SAME cached tuple (lock coalesces fetches).
    first = results[0]
    for r in results[1:]:
        assert r is first or (r[0] is first[0] and r[1] == first[1])
    # CRITICAL: despite 4 concurrent callers, only one Overpass fetch happened.
    assert fake_ox.fetch_count == 1


def test_clear_cache_releases_per_key_locks(monkeypatch):
    """clear_cache must also clear the per-key lock registry so a subsequent
    fetch for the same AOI actually hits osmnx again."""
    fake_ox = _FakeOSMnx(fetch_delay=0.0)
    monkeypatch.setitem(sys.modules, "osmnx", fake_ox)

    nm = NetworkManager()
    aoi = _make_aoi_gdf()
    polygon = aoi.geometry.iloc[0]

    nm.get_graph(aoi, boundary_polygon=polygon)
    assert fake_ox.fetch_count == 1

    nm.clear_cache()
    # Lock registry must be empty after cache clear
    assert nm._key_locks == {}

    nm.get_graph(aoi, boundary_polygon=polygon)
    assert fake_ox.fetch_count == 2
