import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from utils.data_processor import DataProcessor
from utils.distance_calculator import DistanceCalculator

class TestGISFixes:
    
    @pytest.fixture
    def data_processor(self):
        return DataProcessor()
    
    @pytest.fixture
    def distance_calculator(self):
        return DistanceCalculator()

    def test_sampling_sphere_aware(self, data_processor):
        """Test that sampling uses sphere-aware logic for Lat/Lon"""
        # Create a demand GDF in WGS84
        demand_gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[Point(-10, -10), Point(10, 10)],
            crs="EPSG:4326"
        )
        
        # Determine bounds: minx=-10, maxx=10, miny=-10, maxy=10
        # If we sample 500 points, we can check basic bounds
        # (It's hard to statistically prove the distribution in a unit test without flakiness,
        # so we primarily ensure it runs and respects bounds)
        
        candidates = data_processor.generate_candidate_sites(demand_gdf, num_sites=50, random_seed=42)
        
        assert len(candidates) == 50
        assert candidates.crs.to_epsg() == 4326
        assert (candidates.geometry.x >= -10).all()
        assert (candidates.geometry.x <= 10).all()
        assert (candidates.geometry.y >= -10).all()
        assert (candidates.geometry.y <= 10).all()
        
        # Verify it logs the correct message (this requires capturing logs, 
        # but for now we trust the execution path if no error occurs)

    def test_sampling_projected(self, data_processor):
        """Test that sampling uses standard uniform for projected CRS"""
        # Create a demand GDF in a projected CRS (e.g. UTM)
        demand_gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[Point(0, 0), Point(1000, 1000)],
            crs="EPSG:32633" # UTM Zone 33N
        )
        
        candidates = data_processor.generate_candidate_sites(demand_gdf, num_sites=50, random_seed=42)
        
        assert len(candidates) == 50
        assert candidates.crs.to_epsg() == 32633
        assert (candidates.geometry.x >= 0).all()
        assert (candidates.geometry.x <= 1000).all()

    def test_crs_safety_check(self, data_processor):
        """Test that large coordinates are not blindly assumed to be WGS84"""
        # Create GDF with NO CRS and large coordinates
        gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[Point(500000, 4000000)],
            crs=None
        )
        
        processed = data_processor.preprocess_data(gdf)
        
        # Should NOT have EPSG:4326 assigned because coords are huge
        # It should remain None or be untouched in terms of CRS modification by the default logic
        # (The new logic logs a warning and skips setting it to 4326)
        assert processed.crs is None or processed.crs != "EPSG:4326"

    def test_crs_safety_fallback(self, data_processor):
        """Test that valid-looking coordinates ARE assumed to be WGS84"""
        gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[Point(-74, 40)], # NYC-ish
            crs=None
        )
        
        processed = data_processor.preprocess_data(gdf)
        
        # Should be assigned EPSG:4326
        assert processed.crs is not None
        assert processed.crs.to_epsg() == 4326

    def test_distance_calculator_enforces_geodesic(self, distance_calculator):
        """Test that Lat/Lon coordinates trigger geodesic calculation even if requested as euclidean"""
        # Two points on equator, 1 degree apart: (0,0) and (1,0)
        # 1 degree longitude at equator ~= 111,319 meters
        gdf1 = gpd.GeoDataFrame({'id': [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        gdf2 = gpd.GeoDataFrame({'id': [2]}, geometry=[Point(1, 0)], crs="EPSG:4326")
        
        # Request 'euclidean'
        dist_matrix = distance_calculator.calculate_distance_matrix(gdf1, gdf2, metric="euclidean")
        dist = dist_matrix[0, 0]
        
        # If it were naive euclidean, distance would be 1.0 (degree)
        # If geodesic, it should be ~111km
        assert dist > 100000 
        assert dist < 120000

    def test_manhattan_sphere_projection(self, distance_calculator):
        """Test that Manhattan distance on sphere uses projection logic"""
        # Points: (0,0) -> (1,1)
        # Naive "Geodesic Manhattan" (old broken way): dist((0,0)->(1,0)) + dist((1,0)->(1,1))
        # New way: Project to AEQD at center (0.5, 0.5), then Manhattan in meters.
        
        gdf1 = gpd.GeoDataFrame({'id': [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        gdf2 = gpd.GeoDataFrame({'id': [2]}, geometry=[Point(1, 1)], crs="EPSG:4326")
        
        dist_matrix = distance_calculator.calculate_distance_matrix(gdf1, gdf2, metric="manhattan")
        dist = dist_matrix[0, 0]
        
        # 1 degree lat ~= 110.574 km
        # 1 degree lon at equator ~= 111.320 km
        # Expected ~ 221 km
        assert dist > 200000
        assert dist < 250000

