import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, BinaryIO, Dict, Any
import json
import tempfile
import zipfile
import logging
import numpy as np
from io import BytesIO

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

        # Parse JSON safely and construct GeoDataFrame from features
        try:
            geojson_obj = json.loads(content)
        except Exception as exc:
            logger.error(f"Failed to parse GeoJSON: {exc}")
            raise

        # Build GeoDataFrame from features when possible
        try:
            if 'features' in geojson_obj:
                gdf = gpd.GeoDataFrame.from_features(geojson_obj['features'])
            else:
                # Fallback: write to a temporary file for geopandas to read
                with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False, mode='w', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    gdf = gpd.read_file(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
        except Exception as exc:
            logger.error(f"Failed creating GeoDataFrame from GeoJSON: {exc}")
            raise

        # Ensure CRS is set; prefer embedded CRS if present
        if gdf.crs is None:
            # Try to infer CRS from GeoJSON if present
            crs_from_geojson = None
            try:
                crs_from_geojson = geojson_obj.get('crs')
            except Exception:
                crs_from_geojson = None
            if crs_from_geojson:
                try:
                    # GeoJSON crs may be in legacy format; attempt to parse 'properties.name'
                    name = crs_from_geojson.get('properties', {}).get('name') if isinstance(crs_from_geojson, dict) else None
                    if name:
                        gdf.set_crs(name, inplace=True)
                    else:
                        gdf.set_crs("EPSG:4326", inplace=True)
                except Exception:
                    gdf.set_crs("EPSG:4326", inplace=True)
                    logger.info("Failed to use CRS from GeoJSON; assuming EPSG:4326")
            else:
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

        # Coerce to numeric and drop invalid rows
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        before = len(df)
        df = df.dropna(subset=[lon_col, lat_col])
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with invalid coordinate values in CSV")
        
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
            # Direct .shp file uploads are not supported because sidecar files are required
            raise ValueError("Please upload shapefiles as a .zip containing .shp, .dbf, .shx, and related files")
        
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
            # Check for extreme coordinate values before assuming WGS84
            bounds = gdf.total_bounds
            if bounds[0] < -181 or bounds[2] > 181 or bounds[1] < -91 or bounds[3] > 91:
                logger.warning("Data has no CRS and coordinates outside Lat/Lon range - treating as unknown projected system")
                # Do NOT assign a default CRS that would be incorrect
            else:
                logger.warning("No CRS found but coordinates look like Lat/Lon, assuming EPSG:4326")
                gdf.set_crs("EPSG:4326", inplace=True)
        else:
            try:
                # Compare via EPSG integer when possible
                epsg_code = gdf.crs.to_epsg()
                if epsg_code != 4326:
                    logger.info(f"Converting from {gdf.crs} to EPSG:4326")
                    gdf = gdf.to_crs(4326)
            except Exception:
                # Fallback to string comparison
                if str(gdf.crs) not in ("EPSG:4326", "epsg:4326"):
                    logger.info(f"Converting from {gdf.crs} to EPSG:4326")
                    gdf = gdf.to_crs("EPSG:4326")
        
        # Clean invalid geometries conservatively based on geometry type
        if not gdf.geometry.is_valid.all():
            geom_types = set(gdf.geometry.geom_type.unique())
            if {'Polygon', 'MultiPolygon'} & geom_types:
                logger.warning("Cleaning invalid polygon geometries with buffer(0)")
                gdf.loc[~gdf.geometry.is_valid, 'geometry'] = gdf.loc[~gdf.geometry.is_valid, 'geometry'].buffer(0)
            else:
                # For non-polygon types, drop invalid rows rather than mutate geometry
                invalid_count = (~gdf.geometry.is_valid).sum()
                logger.warning(f"Dropping {invalid_count} invalid non-polygon geometries")
                gdf = gdf[gdf.geometry.is_valid]
        
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
        coerced = pd.to_numeric(gdf[capacity_col], errors='coerce')
        num_invalid = coerced.isna().sum()
        if num_invalid:
            logger.warning(f"Coerced {num_invalid} non-numeric capacity values to NaN in column {capacity_col}; filling with 1.0")
        coerced = coerced.fillna(1.0)
        capacity_data = coerced.astype(float).tolist()
        
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
        coerced = pd.to_numeric(gdf[cost_col], errors='coerce')
        num_invalid = coerced.isna().sum()
        if num_invalid:
            logger.warning(f"Coerced {num_invalid} non-numeric cost values to NaN in column {cost_col}; filling with 0.0")
        coerced = coerced.fillna(0.0)
        cost_data = coerced.astype(float).tolist()
        
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
        coerced = pd.to_numeric(gdf[demand_col], errors='coerce')
        num_invalid = coerced.isna().sum()
        if num_invalid:
            logger.warning(f"Coerced {num_invalid} non-numeric demand values to NaN in column {demand_col}; filling with 0.0")
        coerced = coerced.fillna(0.0)
        demand_data = coerced.astype(float).tolist()
        
        # Validate that all values are non-negative
        if any(val < 0 for val in demand_data):
            logger.warning(f"Found negative demand values in column {demand_col}")
            # Filter out negative values or set them to zero
            demand_data = [max(val, 0.0) for val in demand_data]
        
        logger.info(f"Extracted demand/population data from column '{demand_col}': {len(demand_data)} values")
        return demand_data
    
    def generate_candidate_sites(
        self, 
        demand_gdf: gpd.GeoDataFrame, 
        num_sites: int = 100, 
        random_seed: Optional[int] = None
    ) -> gpd.GeoDataFrame:
        """
        Generate random candidate sites within the extent of demand data.
        For geographic coordinates (lat/lon), uses equal-area sampling on the sphere
        to avoid bias towards poles.
        
        Args:
            demand_gdf: Demand points GeoDataFrame to use for extent
            num_sites: Number of candidate sites to generate (default: 100)
            random_seed: Optional random seed for reproducibility
            
        Returns:
            GeoDataFrame with generated candidate sites
        """
        import numpy as np
        
        if len(demand_gdf) == 0:
            raise ValueError("Demand dataset is empty, cannot generate candidate sites")
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            logger.info(f"Using random seed {random_seed} for candidate site generation")
        
        # Get bounding box of demand data
        bounds = demand_gdf.total_bounds  # minx, miny, maxx, maxy
        minx, miny, maxx, maxy = bounds
        
        # Check if coordinates are geographic (Lat/Lon)
        is_geographic = False
        if demand_gdf.crs and demand_gdf.crs.is_geographic:
            is_geographic = True
        elif demand_gdf.crs is None:
            # Heuristic check
            if -180 <= minx and maxx <= 180 and -90 <= miny and maxy <= 90:
                is_geographic = True
        
        if is_geographic:
            # Use equal-area sampling on sphere for Lat/Lon
            # Longitude is uniform
            random_x = np.random.uniform(minx, maxx, num_sites)
            
            # Latitude needs inverse transform sampling: y = asin(uniform(sin(ymin), sin(ymax)))
            # Convert to radians for calculation
            ymin_rad = np.radians(miny)
            ymax_rad = np.radians(maxy)
            
            # Uniformly sample in sin-space
            sin_vals = np.random.uniform(np.sin(ymin_rad), np.sin(ymax_rad), num_sites)
            
            # Convert back to latitude degrees
            random_y = np.degrees(np.arcsin(sin_vals))
            
            logger.info("Used sphere-aware sampling for geographic coordinates")
        else:
            # Standard uniform sampling for projected coordinates
            random_x = np.random.uniform(minx, maxx, num_sites)
            random_y = np.random.uniform(miny, maxy, num_sites)
        
        # Create GeoDataFrame with generated points
        from shapely.geometry import Point
        
        geometry = [Point(x, y) for x, y in zip(random_x, random_y)]
        
        candidate_gdf = gpd.GeoDataFrame(
            {
                'generated': True,  # Mark as generated
                'site_id': range(num_sites),
                'x': random_x,
                'y': random_y
            },
            geometry=geometry,
            crs=demand_gdf.crs  # Use same CRS as demand data
        )
        
        logger.info(f"Generated {num_sites} candidate sites within demand extent: {bounds}")
        return candidate_gdf
    
    def load_raster_file(self, file: BinaryIO) -> Dict[str, Any]:
        """
        Load raster file (GeoTIFF) and return metadata and processed image for map display.
        
        Args:
            file: Binary file object (GeoTIFF/TIF)
            
        Returns:
            Dictionary containing:
                - 'bounds': [[south, west], [north, east]] in WGS84
                - 'image_bytes': PNG bytes for Folium ImageOverlay
                - 'crs': CRS string
                - 'original_bounds': Original bounds in raster CRS
                - 'filename': Original filename
        """
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            from PIL import Image
        except ImportError as e:
            raise ImportError(f"Required libraries for raster support not installed: {e}. Please install rasterio and Pillow.")
        
        file_name = getattr(file, 'name', 'uploaded_raster.tif')
        
        # Save uploaded file to temporary location for rasterio
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        
        try:
            # Open raster with rasterio
            with rasterio.open(tmp_path) as src:
                # Read raster data
                raster_data = src.read()
                
                # Get CRS and bounds
                src_crs = src.crs
                src_bounds = src.bounds  # left, bottom, right, top
                
                # Transform bounds to WGS84 (EPSG:4326) for Folium
                if src_crs and src_crs.to_string() != 'EPSG:4326':
                    bounds_4326 = transform_bounds(
                        src_crs,
                        'EPSG:4326',
                        src_bounds.left,
                        src_bounds.bottom,
                        src_bounds.right,
                        src_bounds.top
                    )
                    # bounds_4326 is (minx, miny, maxx, maxy)
                    bounds = [
                        [bounds_4326[1], bounds_4326[0]],  # [south, west]
                        [bounds_4326[3], bounds_4326[2]]   # [north, east]
                    ]
                else:
                    # Already in WGS84
                    bounds = [
                        [src_bounds.bottom, src_bounds.left],  # [south, west]
                        [src_bounds.top, src_bounds.right]     # [north, east]
                    ]
                
                # Process raster for visualization
                # Handle multi-band rasters (RGB, RGBA, or single band)
                if raster_data.shape[0] == 1:
                    # Single band - convert to grayscale
                    band = raster_data[0]
                    # Normalize to 0-255
                    band_min = np.nanmin(band)
                    band_max = np.nanmax(band)
                    if band_max > band_min:
                        band_normalized = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                    else:
                        band_normalized = np.zeros_like(band, dtype=np.uint8)
                    # Convert to RGB
                    image_array = np.stack([band_normalized, band_normalized, band_normalized], axis=2)
                elif raster_data.shape[0] == 3:
                    # RGB - normalize each band
                    image_array = np.zeros((raster_data.shape[1], raster_data.shape[2], 3), dtype=np.uint8)
                    for i in range(3):
                        band = raster_data[i]
                        band_min = np.nanmin(band)
                        band_max = np.nanmax(band)
                        if band_max > band_min:
                            band_normalized = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                        else:
                            band_normalized = np.zeros_like(band, dtype=np.uint8)
                        image_array[:, :, i] = band_normalized
                elif raster_data.shape[0] == 4:
                    # RGBA - use first 3 bands for RGB
                    image_array = np.zeros((raster_data.shape[1], raster_data.shape[2], 3), dtype=np.uint8)
                    for i in range(3):
                        band = raster_data[i]
                        band_min = np.nanmin(band)
                        band_max = np.nanmax(band)
                        if band_max > band_min:
                            band_normalized = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                        else:
                            band_normalized = np.zeros_like(band, dtype=np.uint8)
                        image_array[:, :, i] = band_normalized
                else:
                    # More than 4 bands - use first 3
                    logger.warning(f"Raster has {raster_data.shape[0]} bands, using first 3 for RGB visualization")
                    image_array = np.zeros((raster_data.shape[1], raster_data.shape[2], 3), dtype=np.uint8)
                    for i in range(3):
                        band = raster_data[i]
                        band_min = np.nanmin(band)
                        band_max = np.nanmax(band)
                        if band_max > band_min:
                            band_normalized = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                        else:
                            band_normalized = np.zeros_like(band, dtype=np.uint8)
                        image_array[:, :, i] = band_normalized
                
                # Convert numpy array to PIL Image
                # Flip vertically because rasterio uses (0,0) at top-left but image display expects bottom-left
                image_array_flipped = np.flipud(image_array)
                pil_image = Image.fromarray(image_array_flipped, mode='RGB')
                
                # Convert to PNG bytes
                img_bytes_io = BytesIO()
                pil_image.save(img_bytes_io, format='PNG')
                image_bytes = img_bytes_io.getvalue()
                
                result = {
                    'bounds': bounds,
                    'image_bytes': image_bytes,
                    'crs': str(src_crs) if src_crs else 'EPSG:4326',
                    'original_bounds': [src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top],
                    'filename': file_name,
                    'width': src.width,
                    'height': src.height
                }
                
                logger.info(f"Loaded raster file {file_name}: {src.width}x{src.height}, CRS: {src_crs}, bounds: {bounds}")
                return result
                
        finally:
            # Clean up temporary file
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Could not delete temporary raster file {tmp_path}: {e}")

