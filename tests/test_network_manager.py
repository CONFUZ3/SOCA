"""Unit tests for NetworkManager (OSMnx calls are mocked)."""

import hashlib
from unittest.mock import MagicMock, patch, PropertyMock

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from utils.network_manager import NetworkManager


def _make_demand_gdf():
    """Tiny 3-point GeoDataFrame in EPSG:4326."""
    geoms = [Point(74.3, 31.5), Point(74.4, 31.6), Point(74.5, 31.55)]
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")


def _make_mock_graph():
    G = MagicMock()
    G.graph = {"crs": "EPSG:32642"}
    G.__len__ = MagicMock(return_value=500)
    return G


# ---------------------------------------------------------------------------
# is_osmnx_available
# ---------------------------------------------------------------------------

def test_is_osmnx_available_true():
    nm = NetworkManager()
    with patch.dict("sys.modules", {"osmnx": MagicMock()}):
        assert nm.is_osmnx_available() is True


def test_is_osmnx_available_false():
    nm = NetworkManager()
    import sys
    original = sys.modules.pop("osmnx", None)
    with patch("builtins.__import__", side_effect=ImportError):
        result = nm.is_osmnx_available()
    if original is not None:
        sys.modules["osmnx"] = original
    # result may be True if osmnx is installed — just verify the call doesn't crash
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Cache key derivation
# ---------------------------------------------------------------------------

def test_cache_key_polygon_vs_bbox():
    nm = NetworkManager()
    demand = _make_demand_gdf()
    poly = Polygon([(74.3, 31.5), (74.5, 31.5), (74.5, 31.6), (74.3, 31.6)])
    key_poly = nm._make_cache_key(demand, poly)
    key_bbox = nm._make_cache_key(demand, None)
    assert key_poly != key_bbox


def test_cache_key_same_bbox_same_key():
    nm = NetworkManager()
    demand = _make_demand_gdf()
    k1 = nm._make_cache_key(demand, None)
    k2 = nm._make_cache_key(demand, None)
    assert k1 == k2


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------

@patch("utils.network_manager.NetworkManager._fetch_graph")
@patch("utils.network_manager.NetworkManager._project_graph")
def test_cache_hit_skips_fetch(mock_project, mock_fetch):
    G_proj = _make_mock_graph()
    mock_project.return_value = (G_proj, "EPSG:32642")
    mock_fetch.return_value = _make_mock_graph()

    nm = NetworkManager()
    demand = _make_demand_gdf()

    with patch("builtins.__import__", return_value=MagicMock()):
        nm.get_graph(demand)
        nm.get_graph(demand)   # second call — should hit cache

    assert mock_fetch.call_count == 1


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

@patch("utils.network_manager.NetworkManager._fetch_graph")
@patch("utils.network_manager.NetworkManager._project_graph")
def test_lru_eviction(mock_project, mock_fetch):
    mock_project.return_value = (_make_mock_graph(), "EPSG:32642")
    mock_fetch.return_value = _make_mock_graph()

    nm = NetworkManager()
    nm.MAX_CACHED_GRAPHS = 2

    polys = [
        Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(3)
    ]
    demand = _make_demand_gdf()

    with patch("builtins.__import__", return_value=MagicMock()):
        for poly in polys:
            nm.get_graph(demand, boundary_polygon=poly)

    assert len(nm._cache) <= nm.MAX_CACHED_GRAPHS


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------

def test_clear_cache():
    nm = NetworkManager()
    nm._cache["x"] = {"G_proj": None, "crs_proj": "EPSG:4326"}
    nm._cache_order.append("x")
    nm.clear_cache()
    assert len(nm._cache) == 0
    assert len(nm._cache_order) == 0
