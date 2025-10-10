import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, BinaryIO
import json
import tempfile
import zipfile
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles loading, validation, and preprocessing of geospatial data"""
    
    def load_file(self, file: BinaryIO) -> gpd.GeoDataFrame:
        """Load GeoJSON, Shapefile, or CSV with coordinates"""
        # Get file name and extension
        file_name = getattr(file, 'name', 'uploaded_file')
        file_ext = Path(file_name).suffix.lower()
        
        try:
            if file_ext in ['.geojson', '.json']:
                return self._load_geojson(file)
            elif file_ext == '.csv':
                return self._load_csv(file)
            elif file_ext in ['.shp', '.zip']:
                return self._load_shapefile(file)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            logger.error(f"Error loading file {file_name}: {e}")
            raise
    
    def _load_geojson(self, file: BinaryIO) -> gpd.GeoDataFrame:
        """Load GeoJSON file"""
        # Read content
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Parse JSON and create GeoDataFrame
        gdf = gpd.read_file(content)
        
        # Ensure CRS is set
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
            logger.info("No CRS found, assuming EPSG:4326 (WGS84)")
        
        return gdf
    
    def _load_csv(self, file: BinaryIO) -> gpd.GeoDataFrame:
        """Load CSV file with coordinates"""
        # Read CSV
        df = pd.read_csv(file)
        
        # Look for coordinate columns
        coord_cols = self._identify_coordinate_columns(df)
        
        if coord_cols is None:
            raise ValueError("Could not identify coordinate columns in CSV. Expected columns like 'lat/lon', 'latitude/longitude', 'x/y'")
        
        lon_col, lat_col = coord_cols
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326"
        )
        
        return gdf
    
    def _load_shapefile(self, file: BinaryIO) -> gpd.GeoDataFrame:
        """Load Shapefile (possibly zipped)"""
        file_name = getattr(file, 'name', 'uploaded_file')
        
        # If it's a ZIP file, extract it
        if file_name.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save and extract ZIP
                zip_path = Path(tmpdir) / 'upload.zip'
                with open(zip_path, 'wb') as f:
                    f.write(file.read())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                # Find .shp file
                shp_files = list(Path(tmpdir).glob('**/*.shp'))
                if not shp_files:
                    raise ValueError("No .shp file found in ZIP archive")
                
                gdf = gpd.read_file(shp_files[0])
        else:
            # Direct .shp file (need associated files)
            with tempfile.NamedTemporaryFile(suffix='.shp', delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            try:
                gdf = gpd.read_file(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        
        return gdf
    
    def _identify_coordinate_columns(self, df: pd.DataFrame) -> Optional[Tuple[str, str]]:
        """Heuristically identify coordinate columns in DataFrame"""
        columns_lower = {col.lower(): col for col in df.columns}
        
        # Common coordinate column name patterns
        lon_patterns = ['lon', 'longitude', 'long', 'x', 'lng']
        lat_patterns = ['lat', 'latitude', 'y']
        
        lon_col = None
        lat_col = None
        
        for pattern in lon_patterns:
            if pattern in columns_lower:
                lon_col = columns_lower[pattern]
                break
        
        for pattern in lat_patterns:
            if pattern in columns_lower:
                lat_col = columns_lower[pattern]
                break
        
        if lon_col and lat_col:
            return (lon_col, lat_col)
        
        return None
    
    def validate_data(
        self, 
        gdf: gpd.GeoDataFrame, 
        required_fields: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Validate that GeoDataFrame has required fields and valid geometry"""
        # Check if it's empty
        if len(gdf) == 0:
            return False, "Dataset is empty"
        
        # Check for required fields
        missing_fields = [field for field in required_fields if field not in gdf.columns]
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        # Check for null geometries first
        if gdf.geometry.isna().any():
            null_count = gdf.geometry.isna().sum()
            return False, f"Found {null_count} null geometries"
        
        # Check for valid geometry
        if not gdf.geometry.is_valid.all():
            invalid_count = (~gdf.geometry.is_valid).sum()
            return False, f"Found {invalid_count} invalid geometries"
        
        # Check for duplicate geometries
        if gdf.geometry.duplicated().any():
            duplicate_count = gdf.geometry.duplicated().sum()
            logger.warning(f"Found {duplicate_count} duplicate geometries")
        
        # Check for extreme coordinate values
        bounds = gdf.total_bounds
        if bounds[0] < -180 or bounds[2] > 180 or bounds[1] < -90 or bounds[3] > 90:
            logger.warning("Coordinates outside normal lat/lon ranges detected")
        
        return True, None
    
    def identify_data_type(self, gdf: gpd.GeoDataFrame) -> str:
        """Heuristically identify if data is demand_points, candidate_sites, etc."""
        columns_lower = [col.lower() for col in gdf.columns]
        
        # Look for telltale column names
        if any(word in ' '.join(columns_lower) for word in ['demand', 'population', 'need', 'service']):
            return "demand_points"
        elif any(word in ' '.join(columns_lower) for word in ['candidate', 'site', 'facility', 'location']):
            return "candidate_sites"
        elif any(word in ' '.join(columns_lower) for word in ['boundary', 'border', 'region', 'area']):
            return "boundary"
        else:
            # Default based on geometry type
            geom_type = gdf.geometry.type.iloc[0]
            if geom_type == 'Point':
                return "points"
            elif geom_type in ['Polygon', 'MultiPolygon']:
                return "polygons"
            else:
                return "unknown"
    
    def preprocess_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardize CRS, clean geometries, handle missing values"""
        gdf = gdf.copy()
        
        # Standardize CRS to WGS84 if not already
        if gdf.crs is None:
            logger.warning("No CRS found, assuming EPSG:4326")
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs != "EPSG:4326":
            logger.info(f"Converting from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")
        
        # Clean invalid geometries
        if not gdf.geometry.is_valid.all():
            logger.warning("Cleaning invalid geometries")
            gdf.geometry = gdf.geometry.buffer(0)
        
        # Remove null geometries
        if gdf.geometry.isna().any():
            logger.warning(f"Removing {gdf.geometry.isna().sum()} null geometries")
            gdf = gdf[~gdf.geometry.isna()]
        
        # Reset index
        gdf = gdf.reset_index(drop=True)
        
        return gdf
    
    def add_default_weights(self, gdf: gpd.GeoDataFrame, weight_column: str = 'weight') -> gpd.GeoDataFrame:
        """Add default weight column if not present"""
        gdf = gdf.copy()
        if weight_column not in gdf.columns:
            gdf[weight_column] = 1.0
            logger.info(f"Added default {weight_column} column with value 1.0")
        return gdf
    
    def identify_capacity_columns(self, gdf: gpd.GeoDataFrame) -> List[str]:
        """Identify columns that might represent facility capacities (max demand each facility can serve)"""
        capacity_columns = []
        columns_lower = [col.lower() for col in gdf.columns]
        
        # Look for capacity-related column names (facility serving limits)
        capacity_patterns = [
            'capacity', 'cap', 'max_service', 'max_capacity', 'service_capacity',
            'throughput', 'max_throughput', 'serving_capacity', 'max_load',
            'facility_capacity', 'site_capacity', 'max_demand', 'demand_limit'
        ]
        
        for col in gdf.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in capacity_patterns):
                # Check if the column contains numeric data
                if pd.api.types.is_numeric_dtype(gdf[col]):
                    capacity_columns.append(col)
                    logger.info(f"Identified facility capacity column: {col}")
        
        return capacity_columns
    
    def identify_cost_columns(self, gdf: gpd.GeoDataFrame) -> List[str]:
        """Identify columns that might represent facility costs"""
        cost_columns = []
        columns_lower = [col.lower() for col in gdf.columns]
        
        # Look for cost-related column names
        cost_patterns = [
            'cost', 'price', 'expense', 'budget', 'investment', 'capital',
            'facility_cost', 'open_cost', 'establishment_cost', 'setup_cost'
        ]
        
        for col in gdf.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in cost_patterns):
                # Check if the column contains numeric data
                if pd.api.types.is_numeric_dtype(gdf[col]):
                    cost_columns.append(col)
                    logger.info(f"Identified cost column: {col}")
        
        return cost_columns
    
    def identify_demand_columns(self, gdf: gpd.GeoDataFrame) -> List[str]:
        """Identify columns that might represent demand/weight values (population at each demand point)"""
        demand_columns = []
        columns_lower = [col.lower() for col in gdf.columns]
        
        # Look for demand-related column names (population/demand at each location)
        demand_patterns = [
            'demand', 'population', 'pop', 'weight', 'w', 'people', 'residents',
            'customers', 'users', 'clients', 'visitors', 'traffic', 'volume',
            'demand_value', 'pop_count', 'resident_count', 'service_demand'
        ]
        
        for col in gdf.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in demand_patterns):
                # Check if the column contains numeric data
                if pd.api.types.is_numeric_dtype(gdf[col]):
                    demand_columns.append(col)
                    logger.info(f"Identified demand/population column: {col}")
        
        return demand_columns
    
    def extract_capacity_data(self, gdf: gpd.GeoDataFrame) -> Optional[List[float]]:
        """Extract capacity data from the most appropriate column"""
        capacity_columns = self.identify_capacity_columns(gdf)
        
        if not capacity_columns:
            return None
        
        # Use the first (most likely) capacity column
        capacity_col = capacity_columns[0]
        capacity_data = gdf[capacity_col].astype(float).tolist()
        
        # Validate that all values are positive
        if any(val <= 0 for val in capacity_data):
            logger.warning(f"Found non-positive capacity values in column {capacity_col}")
            # Filter out non-positive values or set them to a default
            capacity_data = [max(val, 1.0) for val in capacity_data]
        
        logger.info(f"Extracted capacity data from column '{capacity_col}': {len(capacity_data)} values")
        return capacity_data
    
    def extract_cost_data(self, gdf: gpd.GeoDataFrame) -> Optional[List[float]]:
        """Extract cost data from the most appropriate column"""
        cost_columns = self.identify_cost_columns(gdf)
        
        if not cost_columns:
            return None
        
        # Use the first (most likely) cost column
        cost_col = cost_columns[0]
        cost_data = gdf[cost_col].astype(float).tolist()
        
        # Validate that all values are non-negative
        if any(val < 0 for val in cost_data):
            logger.warning(f"Found negative cost values in column {cost_col}")
            # Filter out negative values or set them to zero
            cost_data = [max(val, 0.0) for val in cost_data]
        
        logger.info(f"Extracted cost data from column '{cost_col}': {len(cost_data)} values")
        return cost_data
    
    def extract_demand_data(self, gdf: gpd.GeoDataFrame) -> Optional[List[float]]:
        """Extract demand/population data from the most appropriate column"""
        demand_columns = self.identify_demand_columns(gdf)
        
        if not demand_columns:
            return None
        
        # Use the first (most likely) demand column
        demand_col = demand_columns[0]
        demand_data = gdf[demand_col].astype(float).tolist()
        
        # Validate that all values are non-negative
        if any(val < 0 for val in demand_data):
            logger.warning(f"Found negative demand values in column {demand_col}")
            # Filter out negative values or set them to zero
            demand_data = [max(val, 0.0) for val in demand_data]
        
        logger.info(f"Extracted demand/population data from column '{demand_col}': {len(demand_data)} values")
        return demand_data

