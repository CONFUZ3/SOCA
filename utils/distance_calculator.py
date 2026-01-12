import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial.distance import cdist
from pyproj import Geod
from typing import Optional, Any, Dict
import logging
import hashlib

logger = logging.getLogger(__name__)


class DistanceCalculator:
    """Computes distance matrices using various metrics with caching for performance.
    
    Uses geodesic calculations for geographic coordinates (lat/lon) to ensure
    accurate distance measurements regardless of latitude.
    """
    
    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 10  # Limit cache size
        self._geod = Geod(ellps="WGS84")  # WGS84 ellipsoid for geodesic calculations
    
    def _looks_like_lonlat(self, gdf: gpd.GeoDataFrame) -> bool:
        """Detect if coordinates appear to be geographic (lon/lat)."""
        try:
            xs = [geom.x for geom in gdf.geometry]
            ys = [geom.y for geom in gdf.geometry]
            if not xs or not ys:
                return False
            
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            
            # Check if coordinates fall within valid lon/lat bounds
            is_lonlat_bounds = (
                min(xs) >= -180 and max(xs) <= 180 and
                min(ys) >= -90 and max(ys) <= 90
            )
            
            # Additional heuristics to avoid false positives with projected coordinates
            is_reasonable_geographic_extent = (
                x_range < 360 and y_range < 180 and
                not (min(xs) > 0 and max(xs) < 1000 and min(ys) > 0 and max(ys) < 1000)
            )
            
            return is_lonlat_bounds and is_reasonable_geographic_extent
        except Exception:
            return False
    
    def _is_geographic(self, gdf: gpd.GeoDataFrame) -> bool:
        """Check if GeoDataFrame has geographic CRS or appears to be lon/lat."""
        if gdf.crs and gdf.crs.is_geographic:
            return True
        if gdf.crs is None and self._looks_like_lonlat(gdf):
            return True
        return False
    
    def _generate_cache_key(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame, 
        metric: str
    ) -> str:
        """Generate cache key for distance matrix including CRS information."""
        origin_coords = np.array([[geom.x, geom.y] for geom in origins.geometry])
        dest_coords = np.array([[geom.x, geom.y] for geom in destinations.geometry])
        
        # Include CRS in cache key to avoid incorrect cached results
        origin_crs = str(origins.crs) if origins.crs else "None"
        dest_crs = str(destinations.crs) if destinations.crs else "None"
        
        key_data = f"{origin_coords.tobytes()}{dest_coords.tobytes()}{metric}{origin_crs}{dest_crs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _manage_cache(self, key: str, value: np.ndarray):
        """Manage cache size and add new entry."""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = value
    
    def _convert_to_meters(self, threshold: float, unit: Optional[str] = None) -> float:
        """Convert threshold to meters. Requires explicit unit specification.
        
        Args:
            threshold: The threshold value to convert
            unit: Unit specification ('m', 'km', 'miles'). If None, assumes meters with warning.
            
        Returns:
            Threshold value in meters
        """
        if unit is None:
            logger.warning(
                f"No unit specified for threshold {threshold}. Assuming meters. "
                "Specify unit explicitly (e.g., 'km', 'm', 'miles') to avoid ambiguity."
            )
            return threshold
        
        unit = unit.lower().strip()
        if unit in ('m', 'meter', 'meters'):
            return threshold
        elif unit in ('km', 'kilometer', 'kilometers'):
            return threshold * 1000
        elif unit in ('mi', 'mile', 'miles'):
            return threshold * 1609.34
        else:
            raise ValueError(f"Unknown unit: {unit}. Use 'm', 'km', or 'miles'.")
    
    def _geodesic_distance_matrix(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame
    ) -> np.ndarray:
        """Calculate geodesic distances for geographic coordinates (lat/lon).
        
        Uses pyproj.Geod with WGS84 ellipsoid for accurate distance calculations
        that account for Earth's curvature, unlike EPSG:3857 which has significant
        distance distortion at higher latitudes.
        
        Returns:
            Distance matrix in meters with shape (len(origins), len(destinations))
        """
        origin_coords = np.array([[geom.x, geom.y] for geom in origins.geometry])
        dest_coords = np.array([[geom.x, geom.y] for geom in destinations.geometry])
        
        n_origins = len(origin_coords)
        n_dests = len(dest_coords)
        distances = np.zeros((n_origins, n_dests))
        
        for i, (ox, oy) in enumerate(origin_coords):
            for j, (dx, dy) in enumerate(dest_coords):
                # geod.inv returns (forward_azimuth, back_azimuth, distance_in_meters)
                _, _, dist = self._geod.inv(ox, oy, dx, dy)
                distances[i, j] = dist
        
        return distances
    
    def _geodesic_manhattan_distance_matrix(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame
    ) -> np.ndarray:
        """Calculate Manhattan-style distances for geographic coordinates.
        
        Projects data to a local Azimuthal Equidistant projection centered on the 
        dataset centroid to calculate accurate grid distances in meters.
        
        Returns:
            Distance matrix in meters with shape (len(origins), len(destinations))
        """
        # Calculate centroid of all points to center the projection
        all_geoms = pd.concat([origins.geometry, destinations.geometry])
        center_lon = all_geoms.x.mean()
        center_lat = all_geoms.y.mean()
        
        # Define local Azimuthal Equidistant projection
        # This preserves distances from the center point
        proj_str = f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m +no_defs"
        
        # Project origins and destinations
        origins_proj = origins.to_crs(proj_str)
        destinations_proj = destinations.to_crs(proj_str)
        
        # Calculate Manhattan distance in the projected plane
        return self._manhattan_distance(origins_proj, destinations_proj)
    
    def calculate_distance_matrix(
        self,
        origins: gpd.GeoDataFrame,
        destinations: gpd.GeoDataFrame,
        metric: str = "euclidean",
        network_graph: Optional[Any] = None
    ) -> np.ndarray:
        """Calculate distance matrix with caching for performance.
        
        For geographic coordinates (lat/lon), uses geodesic calculations for
        accurate distances. For projected coordinates, uses standard Euclidean
        or Manhattan distance.
        
        Args:
            origins: GeoDataFrame of origin points
            destinations: GeoDataFrame of destination points
            metric: Distance metric ("euclidean", "manhattan", "network")
            network_graph: Optional network graph for network distances
            
        Returns:
            Distance matrix in meters with shape (len(origins), len(destinations))
        
        Metrics:
        - euclidean: Straight-line distance (geodesic for geographic CRS)
        - manhattan: Grid-based distance (L1 norm)
        - network: Road network distance (not implemented, falls back to euclidean)
        """
        # Check cache first
        cache_key = self._generate_cache_key(origins, destinations, metric)
        if cache_key in self._cache:
            logger.debug("Using cached distance matrix")
            return self._cache[cache_key]
        
        # Harmonize CRS where possible
        if origins.crs and destinations.crs and origins.crs != destinations.crs:
            destinations = destinations.to_crs(origins.crs)
        
        # Determine if we should use geodesic calculations
        # Strict check: if CRS is present and geographic -> True
        # Heuristic check: if CRS is missing but looks like lat/lon -> True
        use_geodesic = False
        if origins.crs:
            use_geodesic = origins.crs.is_geographic
        else:
            use_geodesic = self._looks_like_lonlat(origins)
            if not use_geodesic:
                logger.debug("CRS is missing and coordinates do not look like Lat/Lon. Using Cartesian calculations.")
        
        if use_geodesic:
            logger.debug(f"Using Geodesic calculations for {metric} metric (likely Lat/Lon)")
        
        # Calculate distance matrix based on metric and coordinate system
        if metric == "euclidean":
            if use_geodesic:
                result = self._geodesic_distance_matrix(origins, destinations)
            else:
                result = self._euclidean_distance(origins, destinations)
        elif metric == "manhattan":
            if use_geodesic:
                result = self._geodesic_manhattan_distance_matrix(origins, destinations)
            else:
                result = self._manhattan_distance(origins, destinations)
        elif metric == "network":
            if network_graph is None:
                logger.warning("Network graph not provided, falling back to Euclidean distance")
                if use_geodesic:
                    result = self._geodesic_distance_matrix(origins, destinations)
                else:
                    result = self._euclidean_distance(origins, destinations)
            else:
                result = self._network_distance(origins, destinations, network_graph)
        else:
            raise ValueError(f"Unknown distance metric: {metric}. Use 'euclidean', 'manhattan', or 'network'")
        
        # Cache the result
        self._manage_cache(cache_key, result)
        return result
    
    def _euclidean_distance(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame
    ) -> np.ndarray:
        """Vectorized Euclidean distance calculation for projected coordinates."""
        origin_coords = np.array([[geom.x, geom.y] for geom in origins.geometry])
        dest_coords = np.array([[geom.x, geom.y] for geom in destinations.geometry])
        
        distances = cdist(origin_coords, dest_coords, metric='euclidean')
        return distances
    
    # Keep public aliases for backward compatibility
    def euclidean_distance(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame
    ) -> np.ndarray:
        """Euclidean distance calculation. Uses geodesic for geographic CRS."""
        if self._is_geographic(origins):
            return self._geodesic_distance_matrix(origins, destinations)
        return self._euclidean_distance(origins, destinations)
    
    def _manhattan_distance(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame
    ) -> np.ndarray:
        """Manhattan (L1) distance for projected coordinates."""
        origin_coords = np.array([[geom.x, geom.y] for geom in origins.geometry])
        dest_coords = np.array([[geom.x, geom.y] for geom in destinations.geometry])
        
        distances = cdist(origin_coords, dest_coords, metric='cityblock')
        return distances
    
    def manhattan_distance(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame
    ) -> np.ndarray:
        """Manhattan distance calculation. Uses geodesic for geographic CRS."""
        if self._is_geographic(origins):
            return self._geodesic_manhattan_distance_matrix(origins, destinations)
        return self._manhattan_distance(origins, destinations)
    
    def _network_distance(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame, 
        network_graph
    ) -> np.ndarray:
        """Network-based distance using OSMnx (placeholder - not implemented)."""
        try:
            import osmnx as ox
            import networkx as nx
            
            # Placeholder for future implementation
            logger.warning("Network distance calculation not fully implemented yet. Using Euclidean as fallback.")
            return self.euclidean_distance(origins, destinations)
            
        except ImportError:
            logger.error("OSMnx not installed. Install with: pip install osmnx")
            raise
    
    # Keep public alias for backward compatibility
    def network_distance(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame, 
        network_graph
    ) -> np.ndarray:
        """Network distance (not implemented, falls back to Euclidean)."""
        return self._network_distance(origins, destinations, network_graph)
    
    def calculate_coverage_matrix(
        self,
        origins: gpd.GeoDataFrame,
        destinations: gpd.GeoDataFrame,
        threshold: float,
        metric: str = "euclidean",
        unit: Optional[str] = None
    ) -> np.ndarray:
        """Calculate binary coverage matrix.
        
        Args:
            origins: GeoDataFrame of origin points (demand points)
            destinations: GeoDataFrame of destination points (candidate sites)
            threshold: Service radius threshold value
            metric: Distance metric ("euclidean", "manhattan", "network")
            unit: Unit for threshold ('m', 'km', 'miles'). If None, assumes meters with warning.
        
        Returns:
            Binary matrix where 1 indicates destination is within threshold of origin
        """
        distances = self.calculate_distance_matrix(origins, destinations, metric)
        
        # Convert threshold to meters
        threshold_meters = self._convert_to_meters(threshold, unit)
        
        logger.info(f"Coverage threshold: {threshold} {unit or 'meters (assumed)'} = {threshold_meters:.0f} meters")
        
        coverage = (distances <= threshold_meters).astype(int)
        return coverage
    
    def get_unit_info(self, threshold: float, unit: Optional[str] = None) -> Dict[str, Any]:
        """Get information about unit conversion.
        
        Args:
            threshold: The threshold value
            unit: Unit specification ('m', 'km', 'miles')
            
        Returns:
            Dictionary with conversion details
        """
        threshold_meters = self._convert_to_meters(threshold, unit)
        
        return {
            "input_value": threshold,
            "input_unit": unit or "meters (assumed)",
            "converted_meters": threshold_meters,
            "unit_specified": unit is not None
        }
