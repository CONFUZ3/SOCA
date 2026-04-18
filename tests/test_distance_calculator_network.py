"""Unit tests for DistanceCalculator._network_distance() using synthetic graphs."""

import numpy as np
import pytest
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from unittest.mock import patch, MagicMock

from utils.distance_calculator import DistanceCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gdf(points_lonlat):
    return gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in points_lonlat],
        crs="EPSG:4326",
    )


def _make_proj_gdf(points_xy, crs="EPSG:32642"):
    return gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in points_xy],
        crs=crs,
    )


def _build_synthetic_graph():
    """Build a tiny projected graph: A-B-C in a line, 1000 m apart."""
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32642"
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=1000.0, y=0.0)
    G.add_node(3, x=2000.0, y=0.0)
    G.add_edge(1, 2, length=1000.0)
    G.add_edge(2, 1, length=1000.0)
    G.add_edge(2, 3, length=1000.0)
    G.add_edge(3, 2, length=1000.0)
    return G


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_network_distance_shape():
    """Output matrix shape matches (n_origins, n_dests)."""
    dc = DistanceCalculator()
    origins = _make_gdf([(74.30, 31.50), (74.35, 31.55)])
    destinations = _make_gdf([(74.40, 31.60), (74.45, 31.65), (74.50, 31.55)])

    G = _build_synthetic_graph()
    mock_ox = MagicMock()
    mock_ox.distance.nearest_nodes.return_value = [1, 2, 3]

    with patch.dict("sys.modules", {"osmnx": mock_ox}):
        origins_proj = origins.to_crs("EPSG:32642")
        destinations_proj = destinations.to_crs("EPSG:32642")
        mock_ox.distance.nearest_nodes.side_effect = [
            [1, 2],        # origin_nodes
            [1, 2, 3],     # dest_nodes
        ]
        result = dc._network_distance(origins, destinations, (G, "EPSG:32642"))

    assert result.shape == (2, 3)
    assert result.dtype == np.float64


def test_network_distance_no_inf_in_connected_graph():
    """Connected graph should produce no infinite distances."""
    dc = DistanceCalculator()
    G = _build_synthetic_graph()
    origins = _make_gdf([(74.30, 31.50)])
    destinations = _make_gdf([(74.30, 31.50), (74.35, 31.55)])

    mock_ox = MagicMock()
    mock_ox.distance.nearest_nodes.side_effect = [[1], [1, 3]]

    with patch.dict("sys.modules", {"osmnx": mock_ox}):
        result = dc._network_distance(origins, destinations, (G, "EPSG:32642"))

    assert not np.any(np.isinf(result))


def test_network_distance_geodesic_fallback_for_disconnected():
    """Disconnected pairs must be filled with geodesic distances, not inf."""
    dc = DistanceCalculator()
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32642"
    G.add_node(10, x=0.0, y=0.0)
    G.add_node(99, x=500000.0, y=500000.0)  # isolated node

    origins = _make_gdf([(74.30, 31.50)])
    destinations = _make_gdf([(74.50, 31.70)])

    mock_ox = MagicMock()
    mock_ox.distance.nearest_nodes.side_effect = [[10], [99]]

    with patch.dict("sys.modules", {"osmnx": mock_ox}):
        result = dc._network_distance(origins, destinations, (G, "EPSG:32642"))

    assert not np.any(np.isinf(result))
    assert result[0, 0] > 0


def test_calculate_distance_matrix_network_metric():
    """calculate_distance_matrix routes 'network' metric to _network_distance."""
    dc = DistanceCalculator()
    origins = _make_gdf([(74.30, 31.50)])
    destinations = _make_gdf([(74.40, 31.60)])
    fake_matrix = np.array([[5000.0]])

    with patch.object(dc, "_network_distance", return_value=fake_matrix) as mock_nd:
        result = dc.calculate_distance_matrix(
            origins, destinations, metric="network", network_graph=("G", "EPSG:32642")
        )

    mock_nd.assert_called_once()
    np.testing.assert_array_equal(result, fake_matrix)


def test_calculate_distance_matrix_network_no_graph_falls_back():
    """network metric without network_graph falls back to euclidean, no crash."""
    dc = DistanceCalculator()
    origins = _make_gdf([(74.30, 31.50)])
    destinations = _make_gdf([(74.40, 31.60)])

    result = dc.calculate_distance_matrix(origins, destinations, metric="network", network_graph=None)
    assert result.shape == (1, 1)
    assert result[0, 0] > 0


def test_calculate_coverage_matrix_passes_network_graph():
    """network_graph is forwarded from calculate_coverage_matrix to _network_distance."""
    dc = DistanceCalculator()
    origins = _make_gdf([(74.30, 31.50)])
    destinations = _make_gdf([(74.30, 31.50)])
    fake_graph = ("G", "EPSG:32642")
    fake_matrix = np.array([[500.0]])

    with patch.object(dc, "_network_distance", return_value=fake_matrix):
        result = dc.calculate_coverage_matrix(
            origins, destinations,
            threshold=1000,
            metric="network",
            unit="m",
            network_graph=fake_graph,
        )

    assert result[0, 0] == 1  # 500 m < 1000 m threshold


def test_unique_dest_deduplication_reduces_dijkstra_runs():
    """When multiple candidates snap to the same node, only 1 Dijkstra run occurs."""
    dc = DistanceCalculator()
    G = _build_synthetic_graph()
    origins = _make_gdf([(74.30, 31.50), (74.35, 31.55)])
    destinations = _make_gdf([(74.30, 31.50), (74.30, 31.50)])  # both snap to node 1

    mock_ox = MagicMock()
    mock_ox.distance.nearest_nodes.side_effect = [[1, 2], [1, 1]]

    dijkstra_call_count = [0]
    original_dijkstra = nx.single_source_dijkstra_path_length

    def counting_dijkstra(G, source, weight):
        dijkstra_call_count[0] += 1
        return original_dijkstra(G, source, weight=weight)

    with patch.dict("sys.modules", {"osmnx": mock_ox}):
        with patch("networkx.single_source_dijkstra_path_length", side_effect=counting_dijkstra):
            dc._network_distance(origins, destinations, (G, "EPSG:32642"))

    assert dijkstra_call_count[0] == 1  # only 1 unique dest node
