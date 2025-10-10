"""Enhanced test suite for optimization features"""
import unittest
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solvers.registry import problem_registry
from utils.distance_calculator import DistanceCalculator
from utils.data_processor import DataProcessor


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimizations and caching"""
    
    def setUp(self):
        """Create test data"""
        # Create larger dataset for performance testing
        n_demand = 50
        n_candidates = 20
        
        # Generate random points
        np.random.seed(42)
        demand_coords = [Point(np.random.random() * 10, np.random.random() * 10) 
                        for _ in range(n_demand)]
        candidate_coords = [Point(np.random.random() * 10, np.random.random() * 10) 
                           for _ in range(n_candidates)]
        
        self.demand_gdf = gpd.GeoDataFrame(
            {"demand": np.random.randint(10, 100, n_demand)},
            geometry=demand_coords,
            crs="EPSG:4326"
        )
        
        self.candidate_gdf = gpd.GeoDataFrame(
            {"capacity": np.random.randint(100, 500, n_candidates)},
            geometry=candidate_coords,
            crs="EPSG:4326"
        )
    
    def test_distance_calculator_caching(self):
        """Test that distance calculator caching works"""
        calc = DistanceCalculator()
        
        # First calculation
        start_time = time.time()
        dist1 = calc.calculate_distance_matrix(
            self.demand_gdf, self.candidate_gdf, metric="euclidean"
        )
        first_time = time.time() - start_time
        
        # Second calculation (should use cache)
        start_time = time.time()
        dist2 = calc.calculate_distance_matrix(
            self.demand_gdf, self.candidate_gdf, metric="euclidean"
        )
        second_time = time.time() - start_time
        
        # Results should be identical
        np.testing.assert_array_equal(dist1, dist2)
        
        # Second calculation should be faster (cached)
        self.assertLess(second_time, first_time)
        print(f"First calculation: {first_time:.4f}s, Second (cached): {second_time:.4f}s")
    
    def test_mclp_variants_performance(self):
        """Test MCLP variants solve efficiently"""
        solver = problem_registry.get_problem("mclp")
        
        variants_to_test = ["classical", "capacitated", "budget"]
        
        for variant in variants_to_test:
            with self.subTest(variant=variant):
                params = {
                    "n_facilities": 5,
                    "service_radius": 2.0,
                    "variant": variant
                }
                
                # Add variant-specific parameters
                if variant == "capacitated":
                    params["capacities"] = [100] * len(self.candidate_gdf)
                elif variant == "budget":
                    params["budget"] = 1000
                
                start_time = time.time()
                solution = solver.solve(
                    data={
                        "demand_points": self.demand_gdf,
                        "candidate_sites": self.candidate_gdf
                    },
                    parameters=params,
                    constraints={}
                )
                solve_time = time.time() - start_time
                
                self.assertEqual(solution["status"], "optimal")
                self.assertLess(solve_time, 30)  # Should solve within 30 seconds
                print(f"{variant} MCLP solved in {solve_time:.2f}s")


class TestDataProcessingOptimizations(unittest.TestCase):
    """Test data processing optimizations"""
    
    def test_enhanced_validation(self):
        """Test enhanced data validation"""
        processor = DataProcessor()
        
        # Test with valid data
        valid_gdf = gpd.GeoDataFrame(
            {"demand": [1, 2, 3]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326"
        )
        
        is_valid, msg = processor.validate_data(valid_gdf, [])
        self.assertTrue(is_valid)
        self.assertIsNone(msg)
        
        # Test with invalid geometries
        invalid_gdf = gpd.GeoDataFrame(
            {"demand": [1, 2, 3]},
            geometry=[Point(0, 0), Point(1, 1), None],  # None geometry
            crs="EPSG:4326"
        )
        
        is_valid, msg = processor.validate_data(invalid_gdf, [])
        self.assertFalse(is_valid)
        self.assertIn("null geometries", msg)
    
    def test_capacity_detection(self):
        """Test capacity column detection"""
        processor = DataProcessor()
        
        gdf = gpd.GeoDataFrame({
            "name": ["Site1", "Site2"],
            "capacity": [100, 200],
            "max_service": [150, 250],
            "other_col": ["a", "b"]
        }, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
        
        capacity_cols = processor.identify_capacity_columns(gdf)
        self.assertIn("capacity", capacity_cols)
        self.assertIn("max_service", capacity_cols)
        self.assertNotIn("other_col", capacity_cols)


class TestSolverRobustness(unittest.TestCase):
    """Test solver robustness and error handling"""
    
    def setUp(self):
        """Create test data"""
        self.demand_gdf = gpd.GeoDataFrame(
            {"demand": [1, 2, 3]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326"
        )
        
        self.candidate_gdf = gpd.GeoDataFrame(
            {"capacity": [10, 20, 30]},
            geometry=[Point(0.5, 0.5), Point(1.5, 1.5), Point(2.5, 2.5)],
            crs="EPSG:4326"
        )
    
    def test_p_median_edge_cases(self):
        """Test P-Median with edge cases"""
        solver = problem_registry.get_problem("p-median")
        
        # Test with p = number of candidates
        solution = solver.solve(
            data={
                "demand_points": self.demand_gdf,
                "candidate_sites": self.candidate_gdf
            },
            parameters={"n_facilities": 3},  # All candidates
            constraints={}
        )
        
        self.assertEqual(solution["status"], "optimal")
        self.assertEqual(len(solution["selected_facilities"]), 3)
    
    def test_mclp_infeasible_cases(self):
        """Test MCLP with infeasible parameters"""
        solver = problem_registry.get_problem("mclp")
        
        # Test with very small service radius (likely infeasible)
        solution = solver.solve(
            data={
                "demand_points": self.demand_gdf,
                "candidate_sites": self.candidate_gdf
            },
            parameters={
                "n_facilities": 2,
                "service_radius": 0.1  # Very small radius
            },
            constraints={}
        )
        
        # Should handle gracefully
        self.assertIn(solution["status"], ["optimal", "feasible", "infeasible"])


if __name__ == '__main__':
    unittest.main()
