"""Unit tests for confirm_optimization network-fetch block."""

from unittest.mock import MagicMock, patch

import pytest
import geopandas as gpd
from shapely.geometry import Point


def _make_demand_gdf():
    return gpd.GeoDataFrame(
        geometry=[Point(74.3, 31.5), Point(74.4, 31.6)],
        crs="EPSG:4326",
    )


def _pending(metric="network"):
    return {
        "problem_type": "p-median",
        "parameters": {"n_facilities": 2},
        "constraints": {},
        "distance_metric": metric,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_missing_network_manager_returns_error():
    from agent.tools import optimize_tools as ot

    mock_registry = MagicMock()
    mock_registry.get_problem.return_value = MagicMock()

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("network")}

    with (
        patch.object(ot, "get_network_manager", return_value=None),
        patch.object(ot, "get_data", return_value={}),
        patch.object(ot, "get_problem_state", return_value={"parameters": {}, "constraints": {}, "data": {}, "solution_history": []}),
        patch.object(ot, "get_problem_registry", return_value=mock_registry),
    ):
        result = ot.confirm_optimization(ctx)

    assert result["status"] == "error"
    assert "NetworkManager" in result["error_message"]


def test_osmnx_not_installed_returns_error():
    from agent.tools import optimize_tools as ot

    nm = MagicMock()
    nm.is_osmnx_available.return_value = False

    mock_registry = MagicMock()
    mock_registry.get_problem.return_value = MagicMock()

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("network")}

    with (
        patch.object(ot, "get_network_manager", return_value=nm),
        patch.object(ot, "get_data", return_value={}),
        patch.object(ot, "get_problem_state", return_value={"parameters": {}, "constraints": {}, "data": {}, "solution_history": []}),
        patch.object(ot, "get_problem_registry", return_value=mock_registry),
    ):
        result = ot.confirm_optimization(ctx)

    assert result["status"] == "error"
    assert "OSMnx" in result["error_message"]


def test_graph_fetch_failure_returns_error():
    from agent.tools import optimize_tools as ot

    nm = MagicMock()
    nm.is_osmnx_available.return_value = True
    nm.get_graph.side_effect = RuntimeError("timeout")

    demand = _make_demand_gdf()
    data_store = {"demand_points_auto": demand}
    ps = {"parameters": {}, "constraints": {}, "data": data_store, "solution_history": []}

    mock_registry = MagicMock()
    mock_registry.get_problem.return_value = MagicMock()

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("network")}

    with (
        patch.object(ot, "get_network_manager", return_value=nm),
        patch.object(ot, "get_data", return_value=data_store),
        patch.object(ot, "get_problem_state", return_value=ps),
        patch.object(ot, "get_problem_registry", return_value=mock_registry),
    ):
        result = ot.confirm_optimization(ctx)

    assert result["status"] == "error"
    assert "timeout" in result["error_message"]


def test_network_graph_injected_into_data_dict():
    """When fetch succeeds, _network_graph is added to data_dict before solver call."""
    from agent.tools import optimize_tools as ot

    G_proj = MagicMock()
    nm = MagicMock()
    nm.is_osmnx_available.return_value = True
    nm.get_graph.return_value = (G_proj, "EPSG:32642")

    demand = _make_demand_gdf()
    data_store = {"demand_points_auto": demand}
    ps = {"parameters": {}, "constraints": {}, "data": data_store, "solution_history": []}

    captured_data = {}

    def fake_solve(data, parameters, constraints, distance_metric):
        captured_data.update(data)
        return {
            "status": "optimal",
            "selected_facilities": [],
            "objective_value": 0.0,
            "assignments": {},
            "metrics": {},
            "solution_time": 0,
        }

    mock_solver = MagicMock()
    mock_solver.solve.side_effect = fake_solve
    mock_solver.get_required_data.return_value = {}
    mock_registry = MagicMock()
    mock_registry.get_problem.return_value = mock_solver

    processor_mock = MagicMock()
    processor_mock.add_default_weights.side_effect = lambda gdf, **kw: gdf
    processor_mock.generate_candidate_sites.return_value = demand

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("network")}

    with (
        patch.object(ot, "get_network_manager", return_value=nm),
        patch.object(ot, "get_data", return_value=data_store),
        patch.object(ot, "get_problem_state", return_value=ps),
        patch.object(ot, "get_problem_registry", return_value=mock_registry),
        patch("utils.data_processor.DataProcessor", return_value=processor_mock),
        patch.object(ot, "get_generated_sites_count", return_value=2),
        patch.object(ot, "get_generated_sites_seed", return_value=None),
    ):
        ot.confirm_optimization(ctx)

    assert "_network_graph" in captured_data
    assert captured_data["_network_graph"] == (G_proj, "EPSG:32642")
