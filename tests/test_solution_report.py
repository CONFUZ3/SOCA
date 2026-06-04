"""Tests for utils.solution_report.build_analysis_facts."""

import os

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

# Reverse geocoding hits the network — disable for deterministic, offline tests.
os.environ["SOCA_REVERSE_GEOCODE"] = "0"

from utils.solution_report import build_analysis_facts, _weighted_percentile


@pytest.fixture
def case():
    """4 demand points, 3 candidate sites; facilities 0 and 2 selected."""
    demand = gpd.GeoDataFrame(
        {"population": [10.0, 10.0, 30.0, 50.0]},
        geometry=[Point(0, 0), Point(0.01, 0), Point(1, 1), Point(1.01, 1)],
        crs="EPSG:4326",
    )
    candidates = gpd.GeoDataFrame(
        {"id": [0, 1, 2]},
        geometry=[Point(0, 0), Point(0.5, 0.5), Point(1, 1)],
        crs="EPSG:4326",
    )
    # metres: rows = demand, cols = candidates
    D = np.array(
        [
            [100.0, 5000.0, 9000.0],   # near facility 0
            [200.0, 5000.0, 9000.0],   # near facility 0
            [9000.0, 5000.0, 300.0],   # near facility 2
            [9500.0, 5000.0, 8000.0],  # assigned to 2 but far -> a gap
        ]
    )
    solution = {
        "status": "optimal",
        "objective_value": 1234.5,
        "selected_facilities": [0, 2],
        "assignments": {0: 0, 1: 0, 2: 2, 3: 2},
        "metrics": {"objective_name": "total_weighted_distance"},
        "solver": "pulp",
        "solver_details": {"solver": "pulp", "gap": 0.0, "timed_out": False},
        "solver_time_seconds": 1.5,
        "problem_type": "p-median",
        "variant": "base",
        "warnings": ["used euclidean fallback"],
    }
    data = {"demand_points": demand, "candidate_sites": candidates}
    return solution, data, D, demand["population"].to_numpy()


def test_returns_unit_labeled_structure(case):
    solution, data, D, w = case
    facts = build_analysis_facts(
        solution, data, {}, "euclidean", distance_matrix=D, demand_weights=w,
        equity={"gini_coefficient": 0.25, "bottom_decile_avg_distance": 9500.0},
    )
    assert facts is not None
    assert facts["units"]["distance"] == "km"
    assert facts["units"]["distance_metric"] == "euclidean"
    assert facts["scope"]["num_facilities_selected"] == 2
    assert facts["scope"]["num_demand_points"] == 4
    # Distances reported in km, not metres. Pt 3 is assigned to facility 2
    # (8 km), so that — not its 9.5 km distance to facility 0 — is the max.
    assert facts["distance_distribution"]["max_km"] == pytest.approx(8.0, abs=0.01)
    assert facts["distance_distribution"]["min_km"] == pytest.approx(0.1, abs=0.01)


def test_facilities_named_and_sorted_by_demand(case):
    solution, data, D, w = case
    facts = build_analysis_facts(
        solution, data, {}, "euclidean", distance_matrix=D, demand_weights=w,
        equity={"gini_coefficient": 0.25, "bottom_decile_avg_distance": 9500.0},
    )
    facs = facts["facilities"]
    assert len(facs) == 2
    # Facility 2 serves weight 80 (30+50) vs facility 0's 20 -> sorted first.
    assert facs[0]["index"] == 2
    assert facs[0]["demand_served_weight"] == pytest.approx(80.0)
    assert facs[1]["demand_served_weight"] == pytest.approx(20.0)
    # Coordinates present; place is None because reverse geocoding is disabled.
    assert "lat" in facs[0] and "lon" in facs[0]


def test_coverage_gaps_under_radius(case):
    solution, data, D, w = case
    facts = build_analysis_facts(
        solution, data, {"service_radius": 1.0, "service_radius_unit": "km"},
        "euclidean", distance_matrix=D, demand_weights=w,
        equity={"gini_coefficient": 0.25, "bottom_decile_avg_distance": 9500.0},
    )
    cov = facts["coverage"]
    assert cov["service_radius_km"] == pytest.approx(1.0)
    # Demand pts 2 (0.3km) and 0,1 covered; pt 3 (8km) uncovered.
    assert cov["num_uncovered_points"] == 1
    gaps = facts["coverage_gaps"]
    assert len(gaps) == 1
    assert gaps[0]["demand_index"] == 3
    assert gaps[0]["distance_km"] == pytest.approx(8.0, abs=0.01)


def test_equity_ratio_plain_language_inputs(case):
    solution, data, D, w = case
    facts = build_analysis_facts(
        solution, data, {}, "euclidean", distance_matrix=D, demand_weights=w,
        equity={"gini_coefficient": 0.25, "bottom_decile_avg_distance": 9500.0},
    )
    eq = facts["equity"]
    assert eq["gini_coefficient"] == pytest.approx(0.25)
    assert eq["bottom_decile_avg_distance_km"] == pytest.approx(9.5, abs=0.01)
    # ratio = bottom decile / mean, both in km
    assert eq["bottom_decile_vs_mean_ratio"] == pytest.approx(
        eq["bottom_decile_avg_distance_km"] / eq["mean_distance_km"], abs=0.05
    )


def test_returns_none_without_distance_matrix(case):
    solution, data, _D, w = case
    assert build_analysis_facts(solution, data, {}, "euclidean", distance_matrix=None) is None


def test_weighted_percentile_basic():
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    wts = np.array([1.0, 1.0, 1.0, 1.0])
    assert _weighted_percentile(vals, wts, 0.5) == pytest.approx(2.0, abs=1.0)
    assert _weighted_percentile(vals, wts, 1.0) == pytest.approx(4.0)
