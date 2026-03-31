"""Tests for unit consistency in solver metrics"""
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

class TestUnitConsistency(unittest.TestCase):
    """Test that solvers report metrics in the correct units"""
    
    def setUp(self):
        """Create synthetic test data in EPSG:3857 (meters)"""
        # Create demand points in a 100m x 100m grid
        demand_coords = [(i*10, j*10) for i in range(3) for j in range(3)]
        self.demand_gdf = gpd.GeoDataFrame(
            {"demand": [1.0] * 9},
            geometry=[Point(x, y) for x, y in demand_coords],
            crs="EPSG:3857"
        )
        
        # Create candidate sites (corners)
        candidate_coords = [(0, 0), (20, 20)]
        self.candidate_gdf = gpd.GeoDataFrame(
            {"capacity": [10.0, 10.0]},
            geometry=[Point(x, y) for x, y in candidate_coords],
            crs="EPSG:3857"
        )
        
    def test_mclp_feet_consistency(self):
        """MCLP: Test that average distance is reported in feet when unit='ft'"""
        solver = problem_registry.get_problem("mclp")
        # 20m radius is approx 65.6 feet
        # With 20m radius, all points (0-20, 0-20) are covered by (0,0) and (20,20)
        solution = solver.solve(
            data={"demand_points": self.demand_gdf, "candidate_sites": self.candidate_gdf},
            parameters={"n_facilities": 2, "service_radius": 50, "service_radius_unit": "ft"},
            constraints={}
        )
        
        metrics = solution["metrics"]
        print(f"\nDEBUG MCLP Metrics: {metrics}")
        avg_dist = metrics["average_distance_covered"]
        
        # Internal distances are in meters: (0,0), (10,0), (20,0), (0,10), (10,10), (20,10), (0,20), (10,20), (20,20)
        # to nearest of (0,0) or (20,20)
        # Sum = 40.0
        # Avg distance covered = 40 / 6 = 6.666 m
        # 6.666 / 0.3048 = 21.87 feet.
        
        # If it was still in meters, it would be around 6.0
        self.assertGreater(avg_dist, 10.0) # Meters would be < 10

    def test_p_median_km_consistency(self):
        """P-Median: Test that distances are reported in km when unit='km'"""
        solver = problem_registry.get_problem("p-median")
        solution = solver.solve(
            data={"demand_points": self.demand_gdf, "candidate_sites": self.candidate_gdf},
            parameters={"n_facilities": 1, "service_radius_unit": "km"},
            constraints={}
        )
        
        metrics = solution["metrics"]
        avg_dist = metrics["average_distance"]
        
        # Avg dist in meters (to (0,0)): (0 + 10 + 20 + 10 + 14.14 + 22.36 + 20 + 22.36 + 28.28) / 9 = ~16.35 m
        # In KM: 0.01635
        self.assertAlmostEqual(avg_dist, 16.35 / 1000.0, delta=0.01)
        self.assertLess(avg_dist, 1.0)

    def test_p_center_miles_consistency(self):
        """P-Center: Test that max distance is reported in miles when unit='mi'"""
        solver = problem_registry.get_problem("p-center")
        solution = solver.solve(
            data={"demand_points": self.demand_gdf, "candidate_sites": self.candidate_gdf},
            parameters={"n_facilities": 1, "service_radius_unit": "mi"},
            constraints={}
        )
        
        metrics = solution["metrics"]
        max_dist = metrics["max_distance"]
        
        # Max dist in meters (to (0,0)): dist(0,0 to 20,20) = 28.28 m
        # 28.28 meters in miles: 28.28 / 1609.344 = 0.0175 miles
        self.assertAlmostEqual(max_dist, 28.28 / 1609.344, delta=0.01)

    def test_lscp_yards_consistency(self):
        """LSCP: Test that max distance is reported in yards when unit='yd'"""
        solver = problem_registry.get_problem("lscp")
        # 30m is approx 33 yards
        solution = solver.solve(
            data={"demand_points": self.demand_gdf, "candidate_sites": self.candidate_gdf},
            parameters={"service_radius": 33, "service_radius_unit": "yd"},
            constraints={}
        )
        
        metrics = solution["metrics"]
        max_dist = metrics["max_distance"]
        
        # Max dist in meters should be <= 28.28 (since radius covers all)
        # In yards it should be (meters / 0.9144)
        meters_max = 28.28 / 2 # If 2 facilities selected, max dist is 14.14
        if len(solution['selected_facilities']) == 1:
            meters_max = 28.28
            
        self.assertAlmostEqual(max_dist, meters_max / 0.9144, delta=1.0)

if __name__ == '__main__':
    unittest.main()
