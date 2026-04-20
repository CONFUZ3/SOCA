"""Unit tests for confirm_optimization network-fetch block.

Since road-network distance is the default metric, the tool auto-falls back
to geodesic when the road graph cannot be obtained. Callers that require
road-network distance specifically pass strict_network=True to get the old
hard-error behaviour.
"""

from unittest.mock import MagicMock, patch

import geopandas as gpd
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_demand_gdf():
    return gpd.GeoDataFrame(
        geometry=[Point(74.3, 31.5), Point(74.4, 31.6)],
        crs="EPSG:4326",
    )


def _pending(metric="network", strict: bool = False) -> dict:
    return {
        "problem_type": "p-median",
        "parameters": {"n_facilities": 2},
        "constraints": {},
        "distance_metric": metric,
        "strict_network": strict,
    }


def _make_mock_solver_returning_optimal():
    solver = MagicMock()
    solver.solve.return_value = {
        "status": "optimal",
        "selected_facilities": [],
        "objective_value": 0.0,
        "assignments": {},
        "metrics": {},
        "solution_time": 0,
    }
    solver.get_required_data.return_value = {}
    return solver


def _make_registry(solver: MagicMock) -> MagicMock:
    registry = MagicMock()
    registry.get_problem.return_value = solver
    return registry


def _patched_processor():
    processor_mock = MagicMock()
    processor_mock.add_default_weights.side_effect = lambda gdf, **kw: gdf
    processor_mock.generate_candidate_sites.return_value = _make_demand_gdf()
    return processor_mock


# ---------------------------------------------------------------------------
# Auto-fallback tests (default behaviour)
# ---------------------------------------------------------------------------

def test_missing_network_manager_falls_back_with_warning():
    from agent.tools import optimize_tools as ot

    demand = _make_demand_gdf()
    data_store = {"demand_points_auto": demand}
    ps = {"parameters": {}, "constraints": {}, "data": data_store, "solution_history": []}

    solver = _make_mock_solver_returning_optimal()
    registry = _make_registry(solver)

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("network")}

    with (
        patch.object(ot, "get_network_manager", return_value=None),
        patch.object(ot, "get_data", return_value=data_store),
        patch.object(ot, "get_problem_state", return_value=ps),
        patch.object(ot, "get_problem_registry", return_value=registry),
        patch("utils.data_processor.DataProcessor", return_value=_patched_processor()),
        patch.object(ot, "get_generated_sites_count", return_value=2),
        patch.object(ot, "get_generated_sites_seed", return_value=None),
    ):
        result = ot.confirm_optimization(ctx)

    assert result["status"] == "optimal"
    assert result["distance_metric_used"] == "euclidean"
    assert any("NetworkManager" in w or "road-network" in w.lower() for w in result["warnings"])
    # Solver was called with the fallback metric, not "network"
    called_kwargs = solver.solve.call_args.kwargs
    assert called_kwargs["distance_metric"] == "euclidean"
    assert "_network_graph" not in called_kwargs["data"]


def test_osmnx_unavailable_falls_back_with_warning():
    from agent.tools import optimize_tools as ot

    nm = MagicMock()
    nm.is_osmnx_available.return_value = False

    demand = _make_demand_gdf()
    data_store = {"demand_points_auto": demand}
    ps = {"parameters": {}, "constraints": {}, "data": data_store, "solution_history": []}

    solver = _make_mock_solver_returning_optimal()
    registry = _make_registry(solver)

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("network")}

    with (
        patch.object(ot, "get_network_manager", return_value=nm),
        patch.object(ot, "get_data", return_value=data_store),
        patch.object(ot, "get_problem_state", return_value=ps),
        patch.object(ot, "get_problem_registry", return_value=registry),
        patch("utils.data_processor.DataProcessor", return_value=_patched_processor()),
        patch.object(ot, "get_generated_sites_count", return_value=2),
        patch.object(ot, "get_generated_sites_seed", return_value=None),
    ):
        result = ot.confirm_optimization(ctx)

    assert result["status"] == "optimal"
    assert result["distance_metric_used"] == "euclidean"
    assert any("osmnx" in w.lower() for w in result["warnings"])
    # Fetch was never attempted because osmnx is unavailable
    nm.get_graph.assert_not_called()


def test_graph_fetch_exception_falls_back_with_warning():
    from agent.tools import optimize_tools as ot

    nm = MagicMock()
    nm.is_osmnx_available.return_value = True
    nm.get_graph.side_effect = RuntimeError("timeout")

    demand = _make_demand_gdf()
    data_store = {"demand_points_auto": demand}
    ps = {"parameters": {}, "constraints": {}, "data": data_store, "solution_history": []}

    solver = _make_mock_solver_returning_optimal()
    registry = _make_registry(solver)

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("network")}

    with (
        patch.object(ot, "get_network_manager", return_value=nm),
        patch.object(ot, "get_data", return_value=data_store),
        patch.object(ot, "get_problem_state", return_value=ps),
        patch.object(ot, "get_problem_registry", return_value=registry),
        patch("utils.data_processor.DataProcessor", return_value=_patched_processor()),
        patch.object(ot, "get_generated_sites_count", return_value=2),
        patch.object(ot, "get_generated_sites_seed", return_value=None),
    ):
        result = ot.confirm_optimization(ctx)

    assert result["status"] == "optimal"
    assert result["distance_metric_used"] == "euclidean"
    assert any("timeout" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Strict-network tests (opt-in hard-error behaviour)
# ---------------------------------------------------------------------------

def test_strict_network_errors_when_manager_missing():
    from agent.tools import optimize_tools as ot

    mock_registry = MagicMock()
    mock_registry.get_problem.return_value = MagicMock()

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("network", strict=True)}

    with (
        patch.object(ot, "get_network_manager", return_value=None),
        patch.object(ot, "get_data", return_value={}),
        patch.object(
            ot,
            "get_problem_state",
            return_value={"parameters": {}, "constraints": {}, "data": {}, "solution_history": []},
        ),
        patch.object(ot, "get_problem_registry", return_value=mock_registry),
    ):
        result = ot.confirm_optimization(ctx)

    assert result["status"] == "error"
    assert "strict_network" in result["error_message"]


def test_strict_network_errors_when_osmnx_unavailable():
    from agent.tools import optimize_tools as ot

    nm = MagicMock()
    nm.is_osmnx_available.return_value = False

    mock_registry = MagicMock()
    mock_registry.get_problem.return_value = MagicMock()

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("network", strict=True)}

    with (
        patch.object(ot, "get_network_manager", return_value=nm),
        patch.object(ot, "get_data", return_value={}),
        patch.object(
            ot,
            "get_problem_state",
            return_value={"parameters": {}, "constraints": {}, "data": {}, "solution_history": []},
        ),
        patch.object(ot, "get_problem_registry", return_value=mock_registry),
    ):
        result = ot.confirm_optimization(ctx)

    assert result["status"] == "error"
    assert "osmnx" in result["error_message"].lower()


def test_strict_network_errors_when_fetch_fails():
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
    ctx.state = {"pending_optimization": _pending("network", strict=True)}

    with (
        patch.object(ot, "get_network_manager", return_value=nm),
        patch.object(ot, "get_data", return_value=data_store),
        patch.object(ot, "get_problem_state", return_value=ps),
        patch.object(ot, "get_problem_registry", return_value=mock_registry),
    ):
        result = ot.confirm_optimization(ctx)

    assert result["status"] == "error"
    assert "timeout" in result["error_message"]


# ---------------------------------------------------------------------------
# Happy path — network_graph injected into data_dict
# ---------------------------------------------------------------------------

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

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("network")}

    with (
        patch.object(ot, "get_network_manager", return_value=nm),
        patch.object(ot, "get_data", return_value=data_store),
        patch.object(ot, "get_problem_state", return_value=ps),
        patch.object(ot, "get_problem_registry", return_value=mock_registry),
        patch("utils.data_processor.DataProcessor", return_value=_patched_processor()),
        patch.object(ot, "get_generated_sites_count", return_value=2),
        patch.object(ot, "get_generated_sites_seed", return_value=None),
    ):
        result = ot.confirm_optimization(ctx)

    assert "_network_graph" in captured_data
    assert captured_data["_network_graph"] == (G_proj, "EPSG:32642")
    assert result["distance_metric_used"] == "network"
    assert result["warnings"] == []
