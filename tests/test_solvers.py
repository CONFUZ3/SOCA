"""Unit tests for spatial optimization solvers"""
import unittest
from unittest.mock import patch
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solvers.registry import problem_registry
from utils.distance_calculator import DistanceCalculator


class TestPMedianSolver(unittest.TestCase):
    """Test P-Median solver implementation"""
    
    def setUp(self):
        """Create synthetic test data"""
        # Create demand points (5x5 grid)
        demand_coords = [(i, j) for i in range(5) for j in range(5)]
        self.demand_gdf = gpd.GeoDataFrame(
            {"demand": [1.0] * 25},
            geometry=[Point(x, y) for x, y in demand_coords],
            crs="EPSG:3857"
        )
        
        # Create candidate sites (3x3 subset)
        candidate_coords = [(i, j) for i in range(0, 5, 2) for j in range(0, 5, 2)]
        self.candidate_gdf = gpd.GeoDataFrame(
            {"capacity": [10.0] * 9},
            geometry=[Point(x, y) for x, y in candidate_coords],
            crs="EPSG:3857"
        )
        
        self.solver = problem_registry.get_problem("p-median")
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters
        valid, msg = self.solver.validate_parameters({"n_facilities": 3})
        self.assertTrue(valid)
        
        # Invalid parameters
        valid, msg = self.solver.validate_parameters({"n_facilities": 0})
        self.assertFalse(valid)
        self.assertIsNotNone(msg)
        
        # Missing parameters
        valid, msg = self.solver.validate_parameters({})
        self.assertFalse(valid)
    
    def test_solve_simple(self):
        """Test solving simple problem"""
        solution = self.solver.solve(
            data={
                "demand_points": self.demand_gdf,
                "candidate_sites": self.candidate_gdf
            },
            parameters={"n_facilities": 3},
            constraints={},
            distance_metric="euclidean"
        )
        
        self.assertEqual(solution["status"], "optimal")
        self.assertEqual(len(solution["selected_facilities"]), 3)
        self.assertEqual(len(solution["assignments"]), 25)
        self.assertGreater(solution["solution_time"], 0)
        self.assertIn("metrics", solution)

    def test_solve_capacitated(self):
        solver = problem_registry.get_problem("p-median")
        params = {"n_facilities": 3, "variant": "capacitated", "capacities": [10.0]*len(self.candidate_gdf)}
        solution = solver.solve(
            data={
                "demand_points": self.demand_gdf,
                "candidate_sites": self.candidate_gdf
            },
            parameters=params,
            constraints={},
            distance_metric="euclidean"
        )
        self.assertIn(solution["status"], ["optimal", "feasible"])
        self.assertEqual(len(solution["assignments"]), len(self.demand_gdf))

    def test_solve_budget(self):
        solver = problem_registry.get_problem("p-median")
        facility_costs = [1.0 for _ in range(len(self.candidate_gdf))]
        params = {"n_facilities": 3, "variant": "budget", "facility_costs": facility_costs, "budget": 3.0}
        solution = solver.solve(
            data={
                "demand_points": self.demand_gdf,
                "candidate_sites": self.candidate_gdf
            },
            parameters=params,
            constraints={},
            distance_metric="euclidean"
        )
        self.assertIn(solution["status"], ["optimal", "feasible"])
        self.assertEqual(len(solution["selected_facilities"]), 3)

    def test_solve_max_distance(self):
        solver = problem_registry.get_problem("p-median")
        # Set a tight max distance to force localized assignments
        params = {"n_facilities": 3, "variant": "max_distance", "max_assignment_distance": 3.0}
        solution = solver.solve(
            data={
                "demand_points": self.demand_gdf,
                "candidate_sites": self.candidate_gdf
            },
            parameters=params,
            constraints={},
            distance_metric="euclidean"
        )
        # Could be infeasible if mask removes all options for some demand; allow feasible/infeasible
        self.assertIn(solution["status"], ["optimal", "feasible", "infeasible"]) 
    


class TestPCenterSolver(unittest.TestCase):
    """Test P-Center solver implementation"""
    
    def setUp(self):
        """Create synthetic test data"""
        demand_coords = [(i, j) for i in range(5) for j in range(5)]
        self.demand_gdf = gpd.GeoDataFrame(
            {"demand": [1.0] * 25},
            geometry=[Point(x, y) for x, y in demand_coords],
            crs="EPSG:3857"
        )
        
        candidate_coords = [(i, j) for i in range(0, 5, 2) for j in range(0, 5, 2)]
        self.candidate_gdf = gpd.GeoDataFrame(
            {"capacity": [10.0] * 9},
            geometry=[Point(x, y) for x, y in candidate_coords],
            crs="EPSG:3857"
        )
        
        self.solver = problem_registry.get_problem("p-center")
    
    def test_solve(self):
        """Test solving P-Center problem"""
        solution = self.solver.solve(
            data={
                "demand_points": self.demand_gdf,
                "candidate_sites": self.candidate_gdf
            },
            parameters={"n_facilities": 2},
            constraints={}
        )
        
        self.assertEqual(solution["status"], "optimal")
        self.assertEqual(len(solution["selected_facilities"]), 2)
        self.assertIn("max_distance", solution["metrics"])


class TestMCLPSolver(unittest.TestCase):
    """Test MCLP solver implementation"""
    
    def setUp(self):
        """Create synthetic test data"""
        demand_coords = [(i, j) for i in range(5) for j in range(5)]
        self.demand_gdf = gpd.GeoDataFrame(
            {"demand": [1.0] * 25},
            geometry=[Point(x, y) for x, y in demand_coords],
            crs="EPSG:3857"
        )
        
        candidate_coords = [(i, j) for i in range(0, 5, 2) for j in range(0, 5, 2)]
        self.candidate_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in candidate_coords],
            crs="EPSG:3857"
        )
        
        self.solver = problem_registry.get_problem("mclp")
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        valid, msg = self.solver.validate_parameters({
            "n_facilities": 3,
            "service_radius": 2.0
        })
        self.assertTrue(valid)
        
        # Missing service_radius
        valid, msg = self.solver.validate_parameters({"n_facilities": 3})
        self.assertFalse(valid)
    
    def test_solve(self):
        """Test solving MCLP"""
        solution = self.solver.solve(
            data={
                "demand_points": self.demand_gdf,
                "candidate_sites": self.candidate_gdf
            },
            parameters={"n_facilities": 3, "service_radius": 2.0},
            constraints={}
        )
        
        self.assertEqual(solution["status"], "optimal")
        self.assertIn("coverage_percentage", solution["metrics"])


class TestLSCPSolver(unittest.TestCase):
    """Test LSCP solver implementation"""
    
    def setUp(self):
        """Create synthetic test data"""
        demand_coords = [(i, j) for i in range(3) for j in range(3)]
        self.demand_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in demand_coords],
            crs="EPSG:3857"
        )
        
        # Use same points as candidates
        self.candidate_gdf = self.demand_gdf.copy()
        
        self.solver = problem_registry.get_problem("lscp")
    
    def test_solve(self):
        """Test solving LSCP"""
        solution = self.solver.solve(
            data={
                "demand_points": self.demand_gdf,
                "candidate_sites": self.candidate_gdf
            },
            parameters={"service_radius": 1.5},
            constraints={}
        )
        
        self.assertEqual(solution["status"], "optimal")
        self.assertGreater(len(solution["selected_facilities"]), 0)
        self.assertEqual(solution["metrics"]["coverage_percentage"], 100.0)


class TestFacilitySetConstraints(unittest.TestCase):
    """Cross-cutting fixed_open / fixed_closed / existing_facilities."""

    def setUp(self):
        demand_coords = [(i, j) for i in range(5) for j in range(5)]
        self.demand_gdf = gpd.GeoDataFrame(
            {"demand": [1.0] * 25},
            geometry=[Point(x, y) for x, y in demand_coords],
            crs="EPSG:3857",
        )
        candidate_coords = [(i, j) for i in range(0, 5, 2) for j in range(0, 5, 2)]
        self.candidate_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in candidate_coords],
            crs="EPSG:3857",
        )
        self.data = {
            "demand_points": self.demand_gdf,
            "candidate_sites": self.candidate_gdf,
        }

    def _assert_includes_excludes(self, solution, must_include, must_exclude):
        self.assertIn(solution["status"], ["optimal", "feasible"])
        selected = set(solution["selected_facilities"])
        for j in must_include:
            self.assertIn(j, selected, f"fixed_open index {j} missing from selection {selected}")
        for j in must_exclude:
            self.assertNotIn(j, selected, f"fixed_closed index {j} present in selection {selected}")

    def test_p_median_fixed_open_and_closed(self):
        solver = problem_registry.get_problem("p-median")
        solution = solver.solve(
            data=self.data,
            parameters={"n_facilities": 3, "fixed_open": [0], "fixed_closed": [8]},
            constraints={},
            distance_metric="euclidean",
        )
        self._assert_includes_excludes(solution, must_include=[0], must_exclude=[8])

    def test_p_center_existing_facilities(self):
        solver = problem_registry.get_problem("p-center")
        solution = solver.solve(
            data=self.data,
            parameters={"n_facilities": 3, "existing_facilities": [4], "fixed_closed": [0]},
            constraints={},
            distance_metric="euclidean",
        )
        self._assert_includes_excludes(solution, must_include=[4], must_exclude=[0])

    def test_mclp_fixed_open(self):
        solver = problem_registry.get_problem("mclp")
        solution = solver.solve(
            data=self.data,
            parameters={"n_facilities": 3, "service_radius": 2.0, "fixed_open": [2]},
            constraints={},
            distance_metric="euclidean",
        )
        self._assert_includes_excludes(solution, must_include=[2], must_exclude=[])

    def test_lscp_fixed_closed(self):
        solver = problem_registry.get_problem("lscp")
        solution = solver.solve(
            data=self.data,
            parameters={"service_radius": 3.0, "fixed_closed": [0, 8]},
            constraints={},
            distance_metric="euclidean",
        )
        self._assert_includes_excludes(solution, must_include=[], must_exclude=[0, 8])

    def test_validation_rejects_overlap(self):
        solver = problem_registry.get_problem("p-median")
        ok, err = solver.validate_parameters(
            {"n_facilities": 3, "fixed_open": [1], "fixed_closed": [1]}
        )
        self.assertFalse(ok)
        self.assertIn("both", err.lower())

    def test_validation_rejects_non_int(self):
        solver = problem_registry.get_problem("p-center")
        ok, err = solver.validate_parameters(
            {"n_facilities": 2, "fixed_open": ["a"]}
        )
        self.assertFalse(ok)


class TestProblemRegistry(unittest.TestCase):
    """Test problem registry functionality"""

    def test_problem_inference(self):
        """Test problem type inference from text"""
        # Test P-Median detection
        result = problem_registry.infer_problem_type(
            "I need to minimize the average distance to 5 facilities"
        )
        self.assertEqual(result, "p-median")
        
        # Test P-Center detection
        result = problem_registry.infer_problem_type(
            "minimize the maximum distance"
        )
        self.assertEqual(result, "p-center")
        
        # Test MCLP detection
        result = problem_registry.infer_problem_type(
            "maximize coverage within 5km with 3 facilities"
        )
        self.assertEqual(result, "mclp")


class TestDistanceCalculator(unittest.TestCase):
    """Test distance calculator"""
    
    def setUp(self):
        """Create test data"""
        self.origins = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(1, 0), Point(0, 1)],
            crs="EPSG:3857"
        )
        
        self.destinations = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:3857"
        )
        
        self.calc = DistanceCalculator()
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation"""
        dist_matrix = self.calc.calculate_distance_matrix(
            self.origins, self.destinations, metric="euclidean"
        )

        self.assertEqual(dist_matrix.shape, (3, 2))
        self.assertGreaterEqual(dist_matrix.min(), 0)  # Allow zero for coincident points

    def test_coverage_matrix(self):
        """Test coverage matrix calculation"""
        coverage_matrix = self.calc.calculate_coverage_matrix(
            self.origins, self.destinations, threshold=200, metric="euclidean", unit="km"
        )
        
        self.assertEqual(coverage_matrix.shape, (3, 2))
        self.assertTrue(np.all(np.isin(coverage_matrix, [0, 1])))


def _timed_out_mip():
    """A stubbed MIP result that deterministically forces the GA fallback.

    Returning ``timed_out=True`` with an empty incumbent makes ``solve()`` take
    the GA path (no reliance on a tiny MIP budget actually timing out, which
    trivial test instances never do).
    """
    return {
        "status": "feasible",
        "objective_value": 0.0,
        "selected_facilities": [],
        "assignments": {},
        "solver_details": {"solver": "gurobi", "timed_out": True},
    }


def _covering_counts(demand_gdf, candidate_gdf, selected, radius):
    """Per-demand count of *selected* candidates within euclidean ``radius``."""
    d = np.array([(g.x, g.y) for g in demand_gdf.geometry])
    c = np.array([(g.x, g.y) for g in candidate_gdf.geometry])
    counts = []
    for i in range(len(d)):
        n = 0
        for j in selected:
            if np.hypot(d[i, 0] - c[j, 0], d[i, 1] - c[j, 1]) <= radius:
                n += 1
        counts.append(n)
    return counts


class TestPCenterVariants(unittest.TestCase):
    """P-Center weighted and conditional variants."""

    def setUp(self):
        self.solver = problem_registry.get_problem("p-center")
        # Heavy demand at x=0, light demand at x=10.
        self.demand_gdf = gpd.GeoDataFrame(
            {"weight": [100.0, 1.0]},
            geometry=[Point(0, 0), Point(10, 0)],
            crs="EPSG:3857",
        )
        # Candidates at x=0, 5, 10.
        self.candidate_gdf = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(5, 0), Point(10, 0)],
            crs="EPSG:3857",
        )
        self.data = {"demand_points": self.demand_gdf, "candidate_sites": self.candidate_gdf}

    def test_weighted_pulls_toward_heavy_demand(self):
        weighted = self.solver.solve(
            data=self.data,
            parameters={"n_facilities": 1, "variant": "weighted"},
            constraints={},
            distance_metric="euclidean",
        )
        self.assertEqual(weighted["status"], "optimal")
        self.assertEqual(weighted["selected_facilities"], [0])  # the heavy cluster
        self.assertEqual(weighted["metrics"]["objective_name"], "weighted_max_distance")
        self.assertEqual(weighted.get("variant_used"), "weighted")

    def test_vertex_baseline_differs_from_weighted(self):
        vertex = self.solver.solve(
            data=self.data,
            parameters={"n_facilities": 1},  # default variant
            constraints={},
            distance_metric="euclidean",
        )
        self.assertEqual(vertex["status"], "optimal")
        self.assertEqual(vertex["selected_facilities"], [1])  # the midpoint minimax

    def test_conditional_adds_p_beyond_existing(self):
        demand_coords = [(i, j) for i in range(5) for j in range(5)]
        demand_gdf = gpd.GeoDataFrame(
            {"demand": [1.0] * 25},
            geometry=[Point(x, y) for x, y in demand_coords],
            crs="EPSG:3857",
        )
        candidate_coords = [(i, j) for i in range(0, 5, 2) for j in range(0, 5, 2)]
        candidate_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in candidate_coords],
            crs="EPSG:3857",
        )
        solution = self.solver.solve(
            data={"demand_points": demand_gdf, "candidate_sites": candidate_gdf},
            parameters={"n_facilities": 2, "variant": "conditional", "existing_facilities": [4]},
            constraints={},
            distance_metric="euclidean",
        )
        self.assertIn(solution["status"], ["optimal", "feasible"])
        selected = set(solution["selected_facilities"])
        self.assertIn(4, selected)               # existing facility stays open
        self.assertEqual(len(selected), 3)        # 1 existing + 2 new

    def test_validation_rejects_unknown_variant(self):
        ok, _ = self.solver.validate_parameters({"n_facilities": 2, "variant": "absolute"})
        self.assertFalse(ok)

    def test_validation_conditional_requires_existing(self):
        ok, _ = self.solver.validate_parameters({"n_facilities": 2, "variant": "conditional"})
        self.assertFalse(ok)

    def test_weighted_ga_fallback(self):
        # Force the GA path; the heavy cluster (facility 0) should win the
        # weighted minimax and the objective is the weighted max distance.
        with patch.object(self.solver, "_solve_mip", return_value=_timed_out_mip()):
            sol = self.solver.solve(
                data=self.data,
                parameters={"n_facilities": 1, "variant": "weighted",
                            "ga_time_budget_seconds": 5},
                constraints={},
                distance_metric="euclidean",
            )
        self.assertEqual(sol["solver_details"]["solver"], "ga")
        self.assertEqual(sol["status"], "feasible")
        self.assertEqual(len(sol["selected_facilities"]), 1)
        self.assertEqual(sol["selected_facilities"], [0])
        # demand 100@(0,0) served at dist 0, demand 1@(10,0) served at dist 10 => 1*10.
        self.assertAlmostEqual(sol["objective_value"], 10.0, places=6)

    def test_conditional_ga_fallback(self):
        demand_coords = [(i, j) for i in range(5) for j in range(5)]
        demand_gdf = gpd.GeoDataFrame(
            {"demand": [1.0] * 25},
            geometry=[Point(x, y) for x, y in demand_coords],
            crs="EPSG:3857",
        )
        candidate_coords = [(i, j) for i in range(0, 5, 2) for j in range(0, 5, 2)]
        candidate_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in candidate_coords],
            crs="EPSG:3857",
        )
        with patch.object(self.solver, "_solve_mip", return_value=_timed_out_mip()):
            sol = self.solver.solve(
                data={"demand_points": demand_gdf, "candidate_sites": candidate_gdf},
                parameters={"n_facilities": 2, "variant": "conditional",
                            "existing_facilities": [4], "ga_time_budget_seconds": 5},
                constraints={},
                distance_metric="euclidean",
            )
        self.assertEqual(sol["solver_details"]["solver"], "ga")
        self.assertEqual(sol["status"], "feasible")
        selected = set(sol["selected_facilities"])
        self.assertIn(4, selected)            # existing facility stays open
        self.assertEqual(len(selected), 3)    # 1 existing + 2 new


class TestLSCPVariants(unittest.TestCase):
    """LSCP backup, conditional, probabilistic, and partial variants."""

    def setUp(self):
        self.solver = problem_registry.get_problem("lscp")
        demand_coords = [(i, j) for i in range(3) for j in range(3)]
        self.demand_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in demand_coords],
            crs="EPSG:3857",
        )
        self.candidate_gdf = self.demand_gdf.copy()
        self.data = {"demand_points": self.demand_gdf, "candidate_sites": self.candidate_gdf}

    def test_backup_covers_each_demand_k_times(self):
        solution = self.solver.solve(
            data=self.data,
            parameters={"service_radius": 1.5, "variant": "backup", "k_coverage": 2},
            constraints={},
            distance_metric="euclidean",
        )
        self.assertIn(solution["status"], ["optimal", "feasible"])
        counts = _covering_counts(self.demand_gdf, self.candidate_gdf,
                                  solution["selected_facilities"], 1.5)
        self.assertTrue(all(c >= 2 for c in counts), f"coverage counts: {counts}")

    def test_conditional_keeps_existing_open(self):
        solution = self.solver.solve(
            data=self.data,
            parameters={"service_radius": 1.5, "variant": "conditional", "existing_facilities": [0]},
            constraints={},
            distance_metric="euclidean",
        )
        self.assertIn(solution["status"], ["optimal", "feasible"])
        self.assertIn(0, solution["selected_facilities"])
        self.assertEqual(solution["metrics"]["coverage_percentage"], 100.0)

    def test_probabilistic_meets_reliability(self):
        alpha = 0.95
        p = 0.9
        solution = self.solver.solve(
            data=self.data,
            parameters={
                "service_radius": 1.5,
                "variant": "probabilistic",
                "facility_reliability": p,
                "coverage_reliability": alpha,
            },
            constraints={},
            distance_metric="euclidean",
        )
        self.assertIn(solution["status"], ["optimal", "feasible"])
        counts = _covering_counts(self.demand_gdf, self.candidate_gdf,
                                  solution["selected_facilities"], 1.5)
        for c in counts:
            reliability = 1.0 - (1.0 - p) ** c
            self.assertGreaterEqual(reliability + 1e-9, alpha, f"counts={counts}")

    def test_partial_uses_fewer_facilities_than_full(self):
        # radius 0.5 => identity coverage: full cover needs all 9 candidates.
        full = self.solver.solve(
            data=self.data,
            parameters={"service_radius": 0.5},  # base
            constraints={},
            distance_metric="euclidean",
        )
        partial = self.solver.solve(
            data=self.data,
            parameters={"service_radius": 0.5, "variant": "partial", "coverage_fraction": 0.5},
            constraints={},
            distance_metric="euclidean",
        )
        self.assertEqual(full["status"], "optimal")
        self.assertEqual(len(full["selected_facilities"]), 9)
        self.assertIn(partial["status"], ["optimal", "feasible"])
        self.assertLess(len(partial["selected_facilities"]), 9)
        covered = sum(1 for c in _covering_counts(
            self.demand_gdf, self.candidate_gdf, partial["selected_facilities"], 0.5) if c >= 1)
        self.assertGreaterEqual(covered / len(self.demand_gdf), 0.5)

    def test_validation_rejects_unknown_variant(self):
        ok, _ = self.solver.validate_parameters({"service_radius": 1.5, "variant": "nonsense"})
        self.assertFalse(ok)

    def test_validation_conditional_requires_existing(self):
        ok, _ = self.solver.validate_parameters({"service_radius": 1.5, "variant": "conditional"})
        self.assertFalse(ok)

    def test_backup_ga_fallback(self):
        with patch.object(self.solver, "_solve_mip", return_value=_timed_out_mip()):
            sol = self.solver.solve(
                data=self.data,
                parameters={"service_radius": 1.5, "variant": "backup",
                            "k_coverage": 2, "ga_time_budget_seconds": 5},
                constraints={},
                distance_metric="euclidean",
            )
        self.assertEqual(sol["solver_details"]["solver"], "ga")
        self.assertEqual(sol["status"], "feasible")
        counts = _covering_counts(self.demand_gdf, self.candidate_gdf,
                                  sol["selected_facilities"], 1.5)
        self.assertTrue(all(c >= 2 for c in counts), f"coverage counts: {counts}")

    def test_partial_ga_fallback(self):
        with patch.object(self.solver, "_solve_mip", return_value=_timed_out_mip()):
            sol = self.solver.solve(
                data=self.data,
                parameters={"service_radius": 0.5, "variant": "partial",
                            "coverage_fraction": 0.5, "ga_time_budget_seconds": 5},
                constraints={},
                distance_metric="euclidean",
            )
        self.assertEqual(sol["solver_details"]["solver"], "ga")
        # radius 0.5 => identity coverage; full cover needs all 9 candidates.
        self.assertLess(len(sol["selected_facilities"]), 9)
        covered = sum(1 for c in _covering_counts(
            self.demand_gdf, self.candidate_gdf, sol["selected_facilities"], 0.5) if c >= 1)
        self.assertGreaterEqual(covered / len(self.demand_gdf), 0.5)

    def test_probabilistic_ga_fallback(self):
        alpha = 0.95
        p = 0.9
        with patch.object(self.solver, "_solve_mip", return_value=_timed_out_mip()):
            sol = self.solver.solve(
                data=self.data,
                parameters={"service_radius": 1.5, "variant": "probabilistic",
                            "facility_reliability": p, "coverage_reliability": alpha,
                            "ga_time_budget_seconds": 5},
                constraints={},
                distance_metric="euclidean",
            )
        self.assertEqual(sol["solver_details"]["solver"], "ga")
        counts = _covering_counts(self.demand_gdf, self.candidate_gdf,
                                  sol["selected_facilities"], 1.5)
        if sol["status"] == "feasible":
            for c in counts:
                reliability = 1.0 - (1.0 - p) ** c
                self.assertGreaterEqual(reliability + 1e-9, alpha, f"counts={counts}")
        else:
            self.assertEqual(sol["status"], "approximate")
            self.assertTrue(sol.get("warnings"))

    def test_conditional_ga_fallback(self):
        with patch.object(self.solver, "_solve_mip", return_value=_timed_out_mip()):
            sol = self.solver.solve(
                data=self.data,
                parameters={"service_radius": 1.5, "variant": "conditional",
                            "existing_facilities": [0], "ga_time_budget_seconds": 5},
                constraints={},
                distance_metric="euclidean",
            )
        self.assertEqual(sol["solver_details"]["solver"], "ga")
        self.assertEqual(sol["status"], "feasible")
        self.assertIn(0, sol["selected_facilities"])
        self.assertEqual(sol["metrics"]["coverage_percentage"], 100.0)


class TestVariantMetadata(unittest.TestCase):
    """Metadata reflects only implemented variants."""

    def test_removed_variants_absent(self):
        mclp = problem_registry.get_problem("mclp").get_metadata()["variants"]
        self.assertNotIn("hierarchical", mclp)
        self.assertNotIn("dynamic", mclp)

        pcenter = problem_registry.get_problem("p-center").get_metadata()["variants"]
        self.assertEqual(pcenter, ["vertex", "weighted", "conditional"])

        lscp = problem_registry.get_problem("lscp").get_metadata()["variants"]
        self.assertEqual(lscp, ["base", "backup", "conditional", "probabilistic", "partial"])


if __name__ == '__main__':
    unittest.main()

