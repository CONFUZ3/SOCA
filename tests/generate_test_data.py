"""Generate synthetic test data for spatial optimization problems"""
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from pathlib import Path

def generate_test_data(output_dir: Path):
    """Generate sample demand points and candidate sites"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate demand points (5x5 grid)
    print("Generating demand points...")
    demand_coords = []
    demands = []
    names = []
    
    for i in range(5):
        for j in range(5):
            x = -74.0 + i * 0.02  # Longitude (NYC area)
            y = 40.7 + j * 0.02   # Latitude
            demand_coords.append(Point(x, y))
            demands.append(np.random.randint(10, 100))  # Random demand 10-100
            names.append(f"Demand_{i*5+j+1}")
    
    demand_gdf = gpd.GeoDataFrame({
        'name': names,
        'demand': demands,
        'population': [d * 10 for d in demands]  # Population = demand * 10
    }, geometry=demand_coords, crs="EPSG:4326")
    
    demand_path = output_dir / "sample_demand.geojson"
    demand_gdf.to_file(demand_path, driver="GeoJSON")
    print(f"Created {demand_path}")
    
    # Generate candidate sites (3x3 grid, subset of demand grid)
    print("Generating candidate sites...")
    candidate_coords = []
    candidate_names = []
    capacities = []
    costs = []
    
    for i in range(0, 5, 2):  # Every other point
        for j in range(0, 5, 2):
            x = -74.0 + i * 0.02
            y = 40.7 + j * 0.02
            candidate_coords.append(Point(x, y))
            candidate_names.append(f"Site_{len(candidate_names)+1}")
            capacities.append(np.random.randint(100, 500))
            costs.append(np.random.randint(50000, 200000))
    
    candidate_gdf = gpd.GeoDataFrame({
        'name': candidate_names,
        'capacity': capacities,
        'cost': costs
    }, geometry=candidate_coords, crs="EPSG:4326")
    
    candidate_path = output_dir / "sample_candidates.geojson"
    candidate_gdf.to_file(candidate_path, driver="GeoJSON")
    print(f"Created {candidate_path}")
    
    # Generate a larger dataset for performance testing
    print("Generating large test dataset...")
    n_demand = 100
    n_candidates = 50
    
    # Random points within a bounding box
    np.random.seed(42)
    large_demand_coords = [
        Point(
            -74.0 + np.random.random() * 0.2,
            40.7 + np.random.random() * 0.2
        ) for _ in range(n_demand)
    ]
    
    large_demand_gdf = gpd.GeoDataFrame({
        'name': [f"Demand_{i+1}" for i in range(n_demand)],
        'demand': np.random.randint(10, 200, n_demand),
        'population': np.random.randint(100, 2000, n_demand)
    }, geometry=large_demand_coords, crs="EPSG:4326")
    
    large_demand_path = output_dir / "large_demand.geojson"
    large_demand_gdf.to_file(large_demand_path, driver="GeoJSON")
    print(f"Created {large_demand_path}")
    
    large_candidate_coords = [
        Point(
            -74.0 + np.random.random() * 0.2,
            40.7 + np.random.random() * 0.2
        ) for _ in range(n_candidates)
    ]
    
    large_candidate_gdf = gpd.GeoDataFrame({
        'name': [f"Site_{i+1}" for i in range(n_candidates)],
        'capacity': np.random.randint(500, 2000, n_candidates),
        'cost': np.random.randint(100000, 500000, n_candidates)
    }, geometry=large_candidate_coords, crs="EPSG:4326")
    
    large_candidate_path = output_dir / "large_candidates.geojson"
    large_candidate_gdf.to_file(large_candidate_path, driver="GeoJSON")
    print(f"Created {large_candidate_path}")
    
    print(f"\nTest data generated successfully in {output_dir}")
    print(f"- Small dataset: 25 demand points, 9 candidate sites")
    print(f"- Large dataset: {n_demand} demand points, {n_candidates} candidate sites")

if __name__ == "__main__":
    # Generate test data
    test_data_dir = Path(__file__).parent / "test_data"
    generate_test_data(test_data_dir)

