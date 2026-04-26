"""Tests for wall-clock timeouts, Dijkstra budget fallback, and MCLP GA logging.

These cover the resilience paths that were added to keep long-running
jobs responsive:

* ``confirm_optimization`` — solver call that exceeds
  ``SOLVER_WALL_CLOCK_TIMEOUT`` must return a ``timeout`` status with
  an informative warning, rather than hanging the event loop.
* ``DistanceCalculator._network_distance`` — Dijkstra sweep that
  exceeds ``NETWORK_DIJKSTRA_BUDGET_SECONDS`` must fall back to
  geodesic distance for the remaining destination nodes and emit a
  structured warning.
* ``MCLPSolver.solve`` — when the GA fallback is triggered, the
  correct log lines (``succeeded`` / ``failed`` / ``not available``)
  must be emitted per branch so operators can trace the decision.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_demand_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[Point(74.3, 31.5), Point(74.4, 31.6)],
        crs="EPSG:4326",
    )


def _pending(
    metric: str = "euclidean",
    strict: bool = False,
    problem_type: str = "p-median",
) -> dict:
    return {
        "problem_type": problem_type,
        "parameters": {"n_facilities": 2},
        "constraints": {},
        "distance_metric": metric,
        "strict_network": strict,
    }


def _patched_processor() -> MagicMock:
    processor_mock = MagicMock()
    processor_mock.add_default_weights.side_effect = lambda gdf, **_: gdf
    processor_mock.generate_candidate_sites.return_value = _make_demand_gdf()
    return processor_mock


# ---------------------------------------------------------------------------
# 1) confirm_optimization — wall-clock timeout path
# ---------------------------------------------------------------------------


def test_confirm_optimization_returns_timeout_when_solver_exceeds_budget(
    monkeypatch,
):
    """If the solver itself runs longer than SOLVER_WALL_CLOCK_TIMEOUT
    the tool must return status='timeout' with an actionable warning."""

    from agent.tools import optimize_tools as ot
    from config.settings import settings

    monkeypatch.setattr(settings, "SOLVER_WALL_CLOCK_TIMEOUT", 0.5, raising=False)

    def slow_solve(**_kwargs):
        time.sleep(2.0)
        return {"status": "optimal"}

    mock_solver = MagicMock()
    mock_solver.solve.side_effect = slow_solve
    mock_solver.get_required_data.return_value = {}

    registry = MagicMock()
    registry.get_problem.return_value = mock_solver

    demand = _make_demand_gdf()
    data_store = {"demand_points_auto": demand}
    ps = {
        "parameters": {},
        "constraints": {},
        "data": data_store,
        "solution_history": [],
    }

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("euclidean")}

    with (
        patch.object(ot, "get_network_manager", return_value=None),
        patch.object(ot, "get_data", return_value=data_store),
        patch.object(ot, "get_problem_state", return_value=ps),
        patch.object(ot, "get_problem_registry", return_value=registry),
        patch(
            "utils.data_processor.DataProcessor",
            return_value=_patched_processor(),
        ),
        patch.object(ot, "get_generated_sites_count", return_value=2),
        patch.object(ot, "get_generated_sites_seed", return_value=None),
    ):
        result = ot.confirm_optimization(ctx)

    assert result["status"] == "timeout"
    assert result["distance_metric_used"] == "euclidean"
    assert any("wall-clock" in w.lower() for w in result["warnings"])
    assert result["num_facilities_selected"] == 0


# ---------------------------------------------------------------------------
# 2) Dijkstra budget — geodesic fallback
# ---------------------------------------------------------------------------


def _build_linear_graph() -> nx.MultiDiGraph:
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


def test_network_distance_falls_back_to_geodesic_when_budget_exceeded(
    monkeypatch,
):
    """Force the Dijkstra budget to 0s so every destination falls through
    the fallback path and the matrix still comes back finite."""

    from utils.distance_calculator import DistanceCalculator
    from config.settings import settings

    monkeypatch.setattr(
        settings, "NETWORK_DIJKSTRA_BUDGET_SECONDS", 1e-9, raising=False
    )

    dc = DistanceCalculator()
    G = _build_linear_graph()

    origins = gpd.GeoDataFrame(
        geometry=[Point(74.30, 31.50)],
        crs="EPSG:4326",
    )
    destinations = gpd.GeoDataFrame(
        geometry=[Point(74.40, 31.60), Point(74.50, 31.55)],
        crs="EPSG:4326",
    )

    mock_ox = MagicMock()
    mock_ox.distance.nearest_nodes.side_effect = [[1], [2, 3]]

    with patch.dict("sys.modules", {"osmnx": mock_ox}):
        result = dc._network_distance(origins, destinations, (G, "EPSG:32642"))

    assert result.shape == (1, 2)
    assert not np.any(np.isinf(result))
    # Fallback distance should be positive (real geodesic between far points).
    assert np.all(result > 0)


# ---------------------------------------------------------------------------
# 3) MCLP GA fallback logging
# ---------------------------------------------------------------------------


def _mclp_solver_with_stubbed_variant(
    variant_result: dict,
) -> "tuple[object, dict]":
    """Return an MCLPSolver with its MIP variant path stubbed.

    The stub short-circuits ``_solve_variant`` so we never touch Gurobi /
    PuLP in the unit test; the GA is the real code path.
    """
    from solvers.mclp_solver import MCLPSolver

    solver = MCLPSolver()
    # Avoid the entire _prepare_shared_data pipeline which needs real GDFs.
    shared_data = {
        "demand_gdf": _make_demand_gdf(),
        "candidate_gdf": _make_demand_gdf(),
        "coverage_matrix": np.ones((2, 2), dtype=int),
        "distance_matrix": np.array([[100.0, 200.0], [200.0, 100.0]]),
        "demand_weights": np.array([1.0, 1.0]),
        "service_radius": 1000.0,
        "service_radius_unit": "m",
        "facility_costs": None,
        "capacities": None,
        "unit_info": {},
    }
    solver._prepare_shared_data = MagicMock(return_value=shared_data)  # type: ignore[method-assign]
    solver._solve_variant = MagicMock(return_value=variant_result)  # type: ignore[method-assign]
    return solver, shared_data


def test_mclp_ga_fallback_success_logs_succeeded(caplog):
    """When the MIP times out and GA returns a solution, we expect a
    log line that says ``GA fallback succeeded``."""
    from utils.heuristics.genetic_solver import MCLPGeneticSolver

    timed_out_mip = {
        "status": "feasible",
        "objective_value": 1.0,
        "selected_facilities": [0],
        "assignments": {},
        "z_values": {},
        "solver_details": {"solver": "gurobi", "timed_out": True},
    }
    solver, _shared = _mclp_solver_with_stubbed_variant(timed_out_mip)

    with (
        patch.object(MCLPGeneticSolver, "supports_variant", return_value=True),
        patch.object(
            MCLPGeneticSolver,
            "solve",
            return_value={
                "status": "feasible",
                "objective_value": 2.0,
                "selected_facilities": [1],
                "assignments": {0: 1},
                "solver_details": {"solver": "genetic"},
            },
        ),
    ):
        caplog.set_level("INFO", logger="solvers.mclp_solver")
        result = solver.solve(
            data={
                "demand_points": _make_demand_gdf(),
                "candidate_sites": _make_demand_gdf(),
            },
            parameters={
                "n_facilities": 1,
                "service_radius": 1000.0,
                "variant": "classical",
                "fallback_time_limit_seconds": 0.1,
                "ga_time_budget_seconds": 0.1,
            },
            constraints={},
            distance_metric="euclidean",
        )

    assert result["status"] in ("feasible", "optimal")
    assert any(
        "GA fallback succeeded" in rec.message for rec in caplog.records
    )


def test_mclp_ga_fallback_failure_logs_failed(caplog):
    """When the GA raises, we keep the MIP result and log ``GA fallback
    for variant ... failed`` at ERROR level."""
    from utils.heuristics.genetic_solver import MCLPGeneticSolver

    timed_out_mip = {
        "status": "feasible",
        "objective_value": 1.0,
        "selected_facilities": [0],
        "assignments": {},
        "z_values": {},
        "solver_details": {"solver": "gurobi", "timed_out": True},
    }
    solver, _shared = _mclp_solver_with_stubbed_variant(timed_out_mip)

    with (
        patch.object(MCLPGeneticSolver, "supports_variant", return_value=True),
        patch.object(
            MCLPGeneticSolver,
            "solve",
            side_effect=RuntimeError("boom"),
        ),
    ):
        caplog.set_level("ERROR", logger="solvers.mclp_solver")
        result = solver.solve(
            data={
                "demand_points": _make_demand_gdf(),
                "candidate_sites": _make_demand_gdf(),
            },
            parameters={
                "n_facilities": 1,
                "service_radius": 1000.0,
                "variant": "classical",
                "fallback_time_limit_seconds": 0.1,
                "ga_time_budget_seconds": 0.1,
            },
            constraints={},
            distance_metric="euclidean",
        )

    # MIP result preserved (we did not overwrite on GA failure).
    assert result["selected_facilities"] == [0]
    assert any(
        "GA fallback for variant" in rec.message and "failed" in rec.message
        for rec in caplog.records
    )


def test_mclp_ga_not_available_logs_and_returns_mip(caplog):
    """When the GA doesn't support the variant, the log line ``GA
    fallback not available`` should be emitted and the MIP solution
    returned unchanged."""
    from utils.heuristics.genetic_solver import MCLPGeneticSolver

    timed_out_mip = {
        "status": "feasible",
        "objective_value": 1.0,
        "selected_facilities": [0],
        "assignments": {},
        "z_values": {},
        "solver_details": {"solver": "gurobi", "timed_out": True},
    }
    solver, _shared = _mclp_solver_with_stubbed_variant(timed_out_mip)

    with patch.object(
        MCLPGeneticSolver, "supports_variant", return_value=False
    ):
        caplog.set_level("WARNING", logger="solvers.mclp_solver")
        result = solver.solve(
            data={
                "demand_points": _make_demand_gdf(),
                "candidate_sites": _make_demand_gdf(),
            },
            parameters={
                "n_facilities": 1,
                "service_radius": 1000.0,
                "variant": "classical",
                "fallback_time_limit_seconds": 0.1,
                "ga_time_budget_seconds": 0.1,
            },
            constraints={},
            distance_metric="euclidean",
        )

    assert result["selected_facilities"] == [0]
    assert any(
        "GA fallback not available" in rec.message for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# 5) HTTP 403 is retried (Overpass-style throttling)
# ---------------------------------------------------------------------------


def test_make_request_retries_on_403(monkeypatch):
    """403 from public OSM mirrors signals throttling, not auth denial.
    The HTTP helper must retry instead of raising immediately."""

    import requests
    from utils.fetchers import http as fh

    monkeypatch.setattr(fh, "_MAX_RETRIES", 3, raising=False)
    monkeypatch.setattr(fh, "_RETRY_BASE_DELAY", 0.0, raising=False)

    calls = {"n": 0}

    def fake_get(url, **_kw):
        calls["n"] += 1
        resp = MagicMock()
        if calls["n"] < 3:
            resp.status_code = 403
            err = requests.exceptions.HTTPError(response=resp)
            resp.raise_for_status.side_effect = err
        else:
            resp.status_code = 200
            resp.headers = {}
            resp.raise_for_status.return_value = None
        return resp

    monkeypatch.setattr(fh.requests, "get", fake_get)
    resp = fh.make_request("https://overpass.example/api")
    assert resp.status_code == 200
    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# 6) NetworkManager: concurrent fetches coalesce into a single download
# ---------------------------------------------------------------------------


def test_network_manager_coalesces_concurrent_fetches(monkeypatch):
    """Two NetworkManager instances fetching the same polygon concurrently
    must share a single Overpass download, not race each other."""

    import threading
    from shapely.geometry import Polygon
    from utils import network_manager as nm_mod

    # Reset module-level shared state so the test is deterministic.
    with nm_mod._SHARED_REGISTRY_LOCK:
        nm_mod._SHARED_FETCH_LOCKS.clear()
        nm_mod._SHARED_FETCH_RESULTS.clear()

    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    demand = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")

    fetch_count = {"n": 0}
    fetch_gate = threading.Event()

    def fake_fetch_graph(self, ox, demand_gdf, boundary_polygon):
        fetch_count["n"] += 1
        # Block the first caller so the second has to wait on the lock.
        fetch_gate.wait(timeout=2.0)
        G = MagicMock()
        G.nodes = [0, 1]
        G.edges = [(0, 1)]
        return G

    def fake_project_graph(self, ox, G):
        return G, "EPSG:3857"

    monkeypatch.setattr(
        nm_mod.NetworkManager, "_fetch_graph", fake_fetch_graph, raising=True
    )
    monkeypatch.setattr(
        nm_mod.NetworkManager, "_project_graph", fake_project_graph, raising=True
    )
    # Stub osmnx import so the fetch path is exercised without the real lib.
    import sys
    sys.modules.setdefault("osmnx", MagicMock())

    nm1 = nm_mod.NetworkManager()
    nm2 = nm_mod.NetworkManager()

    results = {}

    def worker(nm, tag):
        results[tag] = nm.get_graph(demand, boundary_polygon=polygon)

    t1 = threading.Thread(target=worker, args=(nm1, "a"))
    t2 = threading.Thread(target=worker, args=(nm2, "b"))
    t1.start()
    # Give t1 a moment to acquire the shared lock and call _fetch_graph.
    time.sleep(0.1)
    t2.start()
    # Release t1 so both can complete.
    fetch_gate.set()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert "a" in results and "b" in results
    # Critical: only one real Overpass fetch even though two instances asked.
    assert fetch_count["n"] == 1, (
        f"Expected 1 coalesced fetch, got {fetch_count['n']}"
    )


# ---------------------------------------------------------------------------
# 7) confirm_optimization: huge-AOI area cutoff
# ---------------------------------------------------------------------------


def test_confirm_optimization_skips_network_fetch_for_huge_aoi(monkeypatch):
    """AOIs above NETWORK_FETCH_MAX_AREA_KM2 should skip the Overpass
    download and fall straight through to euclidean with a warning."""

    from agent.tools import optimize_tools as ot
    from config.settings import settings
    from shapely.geometry import Polygon

    monkeypatch.setattr(
        settings, "NETWORK_FETCH_MAX_AREA_KM2", 1.0, raising=False
    )

    # Solver returns a trivial feasible solution.
    mock_solver = MagicMock()
    mock_solver.solve.return_value = {
        "status": "optimal",
        "objective_value": 0.0,
        "selected_facilities": [0],
        "assignments": {},
        "metrics": {},
        "solution_time": 0.0,
    }
    mock_solver.get_required_data.return_value = {}
    mock_solver.explain_solution.return_value = ""
    mock_solver.get_visualization_config.return_value = {}

    registry = MagicMock()
    registry.get_problem.return_value = mock_solver

    # Large boundary polygon (>1 km² in WGS84 degrees).
    big_poly = Polygon(
        [(0, 0), (2, 0), (2, 2), (0, 2)]  # ~49 000 km² near equator
    )
    boundary = gpd.GeoDataFrame(geometry=[big_poly], crs="EPSG:4326")
    demand = _make_demand_gdf()
    data_store = {"boundary_aoi": boundary, "demand_points_auto": demand}

    ps = {
        "parameters": {},
        "constraints": {},
        "data": data_store,
        "solution_history": [],
    }

    # NetworkManager mock: if get_graph is called, the test fails.
    nm = MagicMock()
    nm.is_osmnx_available.return_value = True
    nm.get_graph.side_effect = AssertionError(
        "get_graph must not be called for huge AOIs"
    )

    ctx = MagicMock()
    ctx.state = {"pending_optimization": _pending("network", strict=False)}

    with (
        patch.object(ot, "get_network_manager", return_value=nm),
        patch.object(ot, "get_data", return_value=data_store),
        patch.object(ot, "get_problem_state", return_value=ps),
        patch.object(ot, "get_problem_registry", return_value=registry),
        patch(
            "utils.data_processor.DataProcessor",
            return_value=_patched_processor(),
        ),
        patch.object(ot, "get_generated_sites_count", return_value=2),
        patch.object(ot, "get_generated_sites_seed", return_value=None),
    ):
        result = ot.confirm_optimization(ctx)

    assert result["status"] == "optimal"
    assert result["distance_metric_used"] == "euclidean"
    assert any(
        "exceeds NETWORK_FETCH_MAX_AREA_KM2" in w or "km²" in w
        for w in result.get("warnings", [])
    )
    nm.get_graph.assert_not_called()
