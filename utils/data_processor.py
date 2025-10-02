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
        
        # Check for valid geometry
        if not gdf.geometry.is_valid.all():
            invalid_count = (~gdf.geometry.is_valid).sum()
            return False, f"Found {invalid_count} invalid geometries"
        
        # Check for null geometries
        if gdf.geometry.isna().any():
            null_count = gdf.geometry.isna().sum()
            return False, f"Found {null_count} null geometries"
        
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

