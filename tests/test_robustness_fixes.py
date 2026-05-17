"""Tests for the ADK + network robustness fixes:

* state_bridge: clear_current_context drops every reference.
* optimize_tools._fetch_network_graph: single retry on transient errors.
* utils.network_manager.prefetch_network_graph: wall-clock timeout.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# state_bridge.clear_current_context
# ---------------------------------------------------------------------------

class TestStateBridgeCleanup:
    def setup_method(self):
        from agent.tools import state_bridge
        state_bridge.clear_current_context()

    def teardown_method(self):
        from agent.tools import state_bridge
        state_bridge.clear_current_context()

    def test_clear_drops_all_attrs(self):
        from agent.tools import state_bridge

        state_bridge.set_current_context(
            data={"k": "v"},
            problem_state={"problem_type": "p-median"},
            problem_registry=object(),
            generated_sites_count=42,
            generated_sites_seed=7,
            network_manager=object(),
        )
        assert state_bridge.get_data() == {"k": "v"}
        assert state_bridge.get_problem_state()["problem_type"] == "p-median"
        assert state_bridge.get_generated_sites_count() == 42
        assert state_bridge.get_network_manager() is not None

        state_bridge.clear_current_context()
        # Getters fall back to defaults when attrs are absent
        assert state_bridge.get_data() == {}
        assert state_bridge.get_problem_state() == {}
        assert state_bridge.get_problem_registry() is None
        assert state_bridge.get_generated_sites_count() == 100  # default
        assert state_bridge.get_generated_sites_seed() is None
        assert state_bridge.get_network_manager() is None


# ---------------------------------------------------------------------------
# Network-fetch retry
# ---------------------------------------------------------------------------

def _make_demand_gdf():
    return gpd.GeoDataFrame(
        geometry=[Point(74.3, 31.5), Point(74.4, 31.6)],
        crs="EPSG:4326",
    )


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


def _patched_processor():
    processor_mock = MagicMock()
    processor_mock.add_default_weights.side_effect = lambda gdf, **kw: gdf
    processor_mock.generate_candidate_sites.return_value = _make_demand_gdf()
    return processor_mock


def _pending(metric="network", strict: bool = False) -> dict:
    return {
        "problem_type": "p-median",
        "parameters": {"n_facilities": 2},
        "constraints": {},
        "distance_metric": metric,
        "strict_network": strict,
    }


class TestNetworkRetry:
    def test_transient_timeout_retries_and_succeeds(self):
        """A single TimeoutError on first attempt must be retried, not surfaced."""
        from agent.tools import optimize_tools as ot

        # Mock graph + crs result
        fake_graph = MagicMock()
        fake_graph.nodes = list(range(10))

        call_count = {"n": 0}

        def get_graph_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise TimeoutError("Overpass blip")
            return fake_graph, "EPSG:3857"

        nm = MagicMock()
        nm.is_osmnx_available.return_value = True
        nm.get_graph.side_effect = get_graph_side_effect

        demand = _make_demand_gdf()
        data_store = {"demand_points_auto": demand}
        ps = {"parameters": {}, "constraints": {}, "data": data_store, "solution_history": []}

        solver = _make_mock_solver_returning_optimal()
        registry = MagicMock()
        registry.get_problem.return_value = solver

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

        # Two attempts total (1 failure + 1 success), and solve happens with network distance
        assert call_count["n"] == 2
        assert result["status"] == "optimal"
        # Solver got the graph, NOT a fallback to euclidean
        called_kwargs = solver.solve.call_args.kwargs
        assert called_kwargs["distance_metric"] == "network"

    def test_persistent_timeout_falls_back_after_retry(self):
        """Two transient errors in a row must exhaust the retry and downgrade."""
        from agent.tools import optimize_tools as ot

        nm = MagicMock()
        nm.is_osmnx_available.return_value = True
        nm.get_graph.side_effect = TimeoutError("stuck")

        demand = _make_demand_gdf()
        data_store = {"demand_points_auto": demand}
        ps = {"parameters": {}, "constraints": {}, "data": data_store, "solution_history": []}

        solver = _make_mock_solver_returning_optimal()
        registry = MagicMock()
        registry.get_problem.return_value = solver

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

        # Two attempts before falling back to euclidean
        assert nm.get_graph.call_count == 2
        assert result["status"] == "optimal"
        assert result["distance_metric_used"] == "euclidean"

    def test_deterministic_error_does_not_retry(self):
        """A non-transient error (ValueError) skips retry and downgrades immediately."""
        from agent.tools import optimize_tools as ot

        nm = MagicMock()
        nm.is_osmnx_available.return_value = True
        nm.get_graph.side_effect = ValueError("malformed AOI")

        demand = _make_demand_gdf()
        data_store = {"demand_points_auto": demand}
        ps = {"parameters": {}, "constraints": {}, "data": data_store, "solution_history": []}

        solver = _make_mock_solver_returning_optimal()
        registry = MagicMock()
        registry.get_problem.return_value = solver

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

        # Only one attempt — deterministic errors are not retried
        assert nm.get_graph.call_count == 1
        assert result["distance_metric_used"] == "euclidean"


# ---------------------------------------------------------------------------
# Prefetch thread wall-clock timeout
# ---------------------------------------------------------------------------

class TestPrefetchTimeout:
    def test_hanging_get_graph_marks_failed_within_budget(self, monkeypatch):
        """A get_graph that never returns must not pin the prefetch thread."""
        from utils import network_manager as nm_mod

        # Shorten the timeout so the test runs quickly
        monkeypatch.setattr(
            nm_mod._soca_settings, "NETWORK_FETCH_TIMEOUT", 1.0, raising=False
        )

        # Build an AOI gdf
        from shapely.geometry import Polygon
        aoi = gpd.GeoDataFrame(
            [{"name": "test"}],
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )

        # NetworkManager mock whose get_graph blocks forever
        manager = MagicMock()
        manager.is_osmnx_available.return_value = True
        manager.get_graph.side_effect = lambda *a, **kw: time.sleep(60)

        session_state: dict = {}
        start = time.perf_counter()
        nm_mod.prefetch_network_graph(manager, aoi, session_state)
        elapsed = time.perf_counter() - start

        assert session_state[nm_mod.NETWORK_STATUS_KEY] == "failed"
        assert "exceeded" in session_state[nm_mod.NETWORK_STATUS_ERROR_KEY].lower()
        # Must complete within timeout + reasonable overhead (≤ 5 s)
        assert elapsed < 5.0, f"prefetch ran {elapsed:.1f}s — timeout not enforced"

    def test_successful_get_graph_marks_ready(self):
        from utils import network_manager as nm_mod
        from shapely.geometry import Polygon

        aoi = gpd.GeoDataFrame(
            [{"name": "test"}],
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )

        fake_graph = MagicMock()
        fake_graph.nodes = list(range(100))
        fake_graph.edges = list(range(200))

        manager = MagicMock()
        manager.is_osmnx_available.return_value = True
        manager.get_graph.return_value = (fake_graph, "EPSG:3857")

        session_state: dict = {}
        nm_mod.prefetch_network_graph(manager, aoi, session_state)

        assert session_state[nm_mod.NETWORK_STATUS_KEY] == "ready"
        assert session_state[nm_mod.NETWORK_STATUS_STATS_KEY]["nodes"] == 100
