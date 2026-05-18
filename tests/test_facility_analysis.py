"""Unit tests for utils.facility_analysis and the analyze tool's resolver."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.facility_analysis import analyze_facilities


def _demand_grid(n: int, weight: float = 1.0) -> gpd.GeoDataFrame:
    """Create n×n demand grid in EPSG:4326 (small lat/lon offsets)."""
    pts = [Point(i * 0.001, j * 0.001) for i in range(n) for j in range(n)]
    return gpd.GeoDataFrame(
        {"population": [weight] * (n * n)},
        geometry=pts,
        crs="EPSG:4326",
    )


def _facilities(coords) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": [f"f{i}" for i in range(len(coords))]},
        geometry=[Point(x, y) for x, y in coords],
        crs="EPSG:4326",
    )


class TestAnalyzeFacilities(unittest.TestCase):
    def test_basic_metrics(self):
        demand = _demand_grid(4)
        # Two facilities at opposite corners.
        facs = _facilities([(0.0, 0.0), (0.003, 0.003)])
        result = analyze_facilities(
            demand_gdf=demand,
            facilities_gdf=facs,
            boundary_gdf=None,
            service_radius_m=500.0,
            distance_metric="euclidean",
        )
        self.assertIn("coverage", result)
        self.assertIn("access", result)
        self.assertIn("density", result)
        self.assertEqual(result["density"]["n_facilities"], 2)
        self.assertEqual(result["density"]["n_demand_points"], 16)
        # Per-point distances are populated.
        self.assertEqual(len(result["per_point_distance_m"]), 16)
        self.assertGreaterEqual(result["coverage"]["pct_demand_covered"], 0.0)
        self.assertLessEqual(result["coverage"]["pct_demand_covered"], 100.0)

    def test_geodesic_fallback_warning(self):
        """No network graph passed → falls back with a warning."""
        demand = _demand_grid(3)
        facs = _facilities([(0.0, 0.0)])
        result = analyze_facilities(
            demand_gdf=demand,
            facilities_gdf=facs,
            boundary_gdf=None,
            service_radius_m=1000.0,
            distance_metric="network",
            network_graph=None,
        )
        self.assertTrue(
            any("geodesic" in w.lower() for w in result["warnings"]),
            f"expected geodesic-fallback warning, got {result['warnings']}",
        )
        self.assertEqual(result["distance_metric_used"], "euclidean")

    def test_empty_demand_raises(self):
        empty = gpd.GeoDataFrame({"population": []}, geometry=[], crs="EPSG:4326")
        facs = _facilities([(0.0, 0.0)])
        with self.assertRaises(ValueError):
            analyze_facilities(empty, facs, None, 500.0)

    def test_empty_facilities_raises(self):
        demand = _demand_grid(2)
        empty = gpd.GeoDataFrame({"name": []}, geometry=[], crs="EPSG:4326")
        with self.assertRaises(ValueError):
            analyze_facilities(demand, empty, None, 500.0)

    def test_invalid_radius_raises(self):
        demand = _demand_grid(2)
        facs = _facilities([(0.0, 0.0)])
        with self.assertRaises(ValueError):
            analyze_facilities(demand, facs, None, 0)


class TestFacilityKeyResolution(unittest.TestCase):
    """Cover the auto-pick / explicit / ambiguity branches in analysis_tools."""

    def _bind(self, data_store: dict):
        """Bind a fake state into the thread-local bridge."""
        from agent.tools import state_bridge
        state_bridge.set_current_context(
            data=data_store,
            problem_state={"data": data_store},
            problem_registry=None,
        )

    def tearDown(self):
        from agent.tools import state_bridge
        state_bridge.clear_current_context()

    def test_no_facility_layer(self):
        from agent.tools.analysis_tools import analyze_existing_facilities
        self._bind({"demand_pop": _demand_grid(2)})
        result = analyze_existing_facilities()
        self.assertIn("error", result)
        self.assertIn("facility", result["error"].lower())

    def test_auto_pick_single_layer(self):
        from agent.tools.analysis_tools import analyze_existing_facilities
        self._bind({
            "boundary_aoi": gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
            "demand_pop": _demand_grid(3),
            "health_facilities_overture": _facilities([(0.0, 0.0)]),
        })
        result = analyze_existing_facilities(distance_metric="euclidean")
        self.assertIsNone(result.get("error"))
        self.assertEqual(result["facility_dataset_key"], "health_facilities_overture")

    def test_ambiguous_multiple_layers(self):
        from agent.tools.analysis_tools import analyze_existing_facilities
        self._bind({
            "demand_pop": _demand_grid(2),
            "health_facilities_a": _facilities([(0.0, 0.0)]),
            "school_facilities_b": _facilities([(0.001, 0.001)]),
        })
        result = analyze_existing_facilities()
        self.assertIn("error", result)
        self.assertIn("multiple", result["error"].lower())

    def test_explicit_key_missing(self):
        from agent.tools.analysis_tools import analyze_existing_facilities
        self._bind({"demand_pop": _demand_grid(2)})
        result = analyze_existing_facilities(facility_dataset_key="nope")
        self.assertIn("error", result)
        self.assertIn("not found", result["error"].lower())

    def test_unknown_unit(self):
        from agent.tools.analysis_tools import analyze_existing_facilities
        self._bind({
            "demand_pop": _demand_grid(2),
            "health_facilities_x": _facilities([(0.0, 0.0)]),
        })
        result = analyze_existing_facilities(service_radius_unit="parsec")
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
