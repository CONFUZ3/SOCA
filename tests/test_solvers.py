"""Unit tests for spatial optimization solvers"""
import unittest
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
            crs="EPSG:4326"
        )
        
        # Create candidate sites (3x3 subset)
        candidate_coords = [(i, j) for i in range(0, 5, 2) for j in range(0, 5, 2)]
        self.candidate_gdf = gpd.GeoDataFrame(
            {"capacity": [10.0] * 9},
            geometry=[Point(x, y) for x, y in candidate_coords],
            crs="EPSG:4326"
        )
        
        self.solver = problem_registry.get_problem("p-median")
    
    def test_metadata(self):
        """Test that metadata is complete"""
        metadata = self.solver.get_metadata()
        self.assertIn("name", metadata)
        self.assertIn("academic_refs", metadata)
        self.assertTrue(len(metadata["academic_refs"]) > 0)
        self.assertEqual(metadata["short_name"], "p-median")
    
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
    
    def test_explanation_generation(self):
        """Test that explanation is generated"""
        solution = self.solver.solve(
            data={
                "demand_points": self.demand_gdf,
                "candidate_sites": self.candidate_gdf
            },
            parameters={"n_facilities": 3},
            constraints={}
        )
        
        explanation = self.solver.explain_solution(
            solution=solution,
            data={
                "demand_points": self.demand_gdf,
                "candidate_sites": self.candidate_gdf
            }
        )
        
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 50)


class TestPCenterSolver(unittest.TestCase):
    """Test P-Center solver implementation"""
    
    def setUp(self):
        """Create synthetic test data"""
        demand_coords = [(i, j) for i in range(5) for j in range(5)]
        self.demand_gdf = gpd.GeoDataFrame(
            {"demand": [1.0] * 25},
            geometry=[Point(x, y) for x, y in demand_coords],
            crs="EPSG:4326"
        )
        
        candidate_coords = [(i, j) for i in range(0, 5, 2) for j in range(0, 5, 2)]
        self.candidate_gdf = gpd.GeoDataFrame(
            {"capacity": [10.0] * 9},
            geometry=[Point(x, y) for x, y in candidate_coords],
            crs="EPSG:4326"
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
            crs="EPSG:4326"
        )
        
        candidate_coords = [(i, j) for i in range(0, 5, 2) for j in range(0, 5, 2)]
        self.candidate_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in candidate_coords],
            crs="EPSG:4326"
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
            crs="EPSG:4326"
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


class TestProblemRegistry(unittest.TestCase):
    """Test problem registry functionality"""
    
    def test_problem_registration(self):
        """Test that problems are registered"""
        problems = problem_registry.list_problems()
        self.assertGreater(len(problems), 0)
        
        # Check for expected problems
        short_names = [p['short_name'] for p in problems]
        self.assertIn('p-median', short_names)
        self.assertIn('p-center', short_names)
        self.assertIn('mclp', short_names)
        self.assertIn('lscp', short_names)
    
    def test_problem_retrieval(self):
        """Test retrieving problem by name"""
        solver = problem_registry.get_problem("p-median")
        self.assertIsNotNone(solver)
        
        metadata = solver.get_metadata()
        self.assertEqual(metadata['short_name'], 'p-median')
    
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
            crs="EPSG:4326"
        )
        
        self.destinations = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326"
        )
        
        self.calc = DistanceCalculator()
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation"""
        dist_matrix = self.calc.calculate_distance_matrix(
            self.origins, self.destinations, metric="euclidean"
        )
        
        self.assertEqual(dist_matrix.shape, (3, 2))
        self.assertGreater(dist_matrix.min(), 0)
    
    def test_manhattan_distance(self):
        """Test Manhattan distance calculation"""
        dist_matrix = self.calc.calculate_distance_matrix(
            self.origins, self.destinations, metric="manhattan"
        )
        
        self.assertEqual(dist_matrix.shape, (3, 2))
    
    def test_coverage_matrix(self):
        """Test coverage matrix calculation"""
        coverage_matrix = self.calc.calculate_coverage_matrix(
            self.origins, self.destinations, threshold=200000, metric="euclidean"
        )
        
        self.assertEqual(coverage_matrix.shape, (3, 2))
        self.assertTrue(np.all(np.isin(coverage_matrix, [0, 1])))


if __name__ == '__main__':
    unittest.main()

