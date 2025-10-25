import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from typing import Optional, Any, Dict, Tuple
import logging
import hashlib
from functools import lru_cache

logger = logging.getLogger(__name__)

class DistanceCalculator:
    """Computes distance matrices using various metrics with caching for performance"""
    
    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 10  # Limit cache size
    
    def _looks_like_lonlat(self, gdf: gpd.GeoDataFrame) -> bool:
        """Improved detection of lat/lon coordinates with better heuristics"""
        try:
            xs = [geom.x for geom in gdf.geometry]
            ys = [geom.y for geom in gdf.geometry]
            if not xs or not ys:
                return False
            
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            
            # More robust lat/lon detection
            is_lonlat_bounds = (
                min(xs) >= -180 and max(xs) <= 180 and
                min(ys) >= -90 and max(ys) <= 90
            )
            
            # Additional heuristics to avoid false positives
            is_reasonable_geographic_extent = (
                x_range < 360 and y_range < 180 and  # Reasonable geographic extent
                not (min(xs) > 0 and max(xs) < 1000 and min(ys) > 0 and max(ys) < 1000)  # Avoid UTM-like coordinates
            )
            
            # Debug output (can be removed in production)
            # logger.debug(f"Lat/lon detection: bounds={is_lonlat_bounds}, extent={is_reasonable_geographic_extent}, "
            #             f"x_range={x_range:.3f}, y_range={y_range:.3f}")
            
            return is_lonlat_bounds and is_reasonable_geographic_extent
        except Exception:
            return False
    
    def _generate_cache_key(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame, 
        metric: str
    ) -> str:
        """Generate cache key for distance matrix"""
        # Create hash from geometry coordinates and metric
        origin_coords = np.array([[geom.x, geom.y] for geom in origins.geometry])
        dest_coords = np.array([[geom.x, geom.y] for geom in destinations.geometry])
        
        key_data = f"{origin_coords.tobytes()}{dest_coords.tobytes()}{metric}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _manage_cache(self, key: str, value: np.ndarray):
        """Manage cache size and add new entry"""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = value
    
    def _smart_unit_conversion(self, threshold: float, gdf: gpd.GeoDataFrame, user_unit_hint: str = None) -> tuple[float, str]:
        """
        Smart unit detection and conversion for service radius.
        Returns (converted_value_in_meters, unit_description)
        """
        # If user provided a unit hint, use it
        if user_unit_hint:
            if user_unit_hint.lower() in ['km', 'kilometers', 'kilometer']:
                meters = threshold * 1000
                return meters, f"{threshold} km -> {meters:.0f} m (user specified kilometers)"
            elif user_unit_hint.lower() in ['m', 'meters', 'meter']:
                return threshold, f"{threshold} m (user specified meters)"
        
        # For ambiguous cases, be conservative and ask user to specify
        # Don't auto-assume units based on coordinate system
        if self._looks_like_lonlat(gdf):
            # For lat/lon data, be conservative - don't assume units
            # Return meters as default but flag for user confirmation
            return threshold, f"{threshold} m (default - please confirm units for lat/lon data)"
        else:
            # For projected coordinates, meters is usually correct
            return threshold, f"{threshold} m (projected coordinates detected)"
    
    def _suggest_units_for_user(self, threshold: float, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Suggest appropriate units to the user based on data characteristics.
        Returns suggestions for user confirmation.
        """
        # Analyze the data to make intelligent suggestions
        bounds = gdf.total_bounds
        x_range = bounds[2] - bounds[0]
        y_range = bounds[3] - bounds[1]
        max_extent = max(x_range, y_range)
        
        suggestions = {
            "threshold": threshold,
            "data_extent_km": max_extent * 111 if self._looks_like_lonlat(gdf) else max_extent / 1000,  # Rough conversion
            "suggestions": []
        }
        
        # For lat/lon data, be neutral and ask user to specify
        if self._looks_like_lonlat(gdf):
            suggestions["suggestions"].append({
                "unit": "meters",
                "value": threshold,
                "converted_meters": threshold,
                "reason": "Most common for service radius values (e.g., 3000m = 3km)",
                "recommended": True
            })
            suggestions["suggestions"].append({
                "unit": "kilometers", 
                "value": threshold,
                "converted_meters": threshold * 1000,
                "reason": "If you meant a very large service area (e.g., 3000km)",
                "recommended": False
            })
        else:
            suggestions["suggestions"].append({
                "unit": "meters",
                "value": threshold,
                "converted_meters": threshold,
                "reason": "Your data appears to be in a projected coordinate system",
                "recommended": True
            })
            suggestions["suggestions"].append({
                "unit": "kilometers",
                "value": threshold,
                "converted_meters": threshold * 1000,
                "reason": "If you meant a very large service area",
                "recommended": False
            })
        
        return suggestions
    
    def _detect_reasonable_threshold(self, threshold: float, gdf: gpd.GeoDataFrame) -> bool:
        """
        Detect if threshold value seems reasonable for the data extent.
        Helps catch unit mistakes (e.g., 3000 km when user meant 3000 m).
        """
        try:
            # Get data bounds
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            x_range = bounds[2] - bounds[0]
            y_range = bounds[3] - bounds[1]
            max_extent = max(x_range, y_range)
            
            # If threshold is much larger than data extent, it might be wrong units
            if gdf.crs and gdf.crs.is_geographic:
                # For geographic data, threshold in km should be reasonable
                if threshold > max_extent * 10:  # 10x data extent seems too large
                    return False
            else:
                # For projected data, threshold in meters should be reasonable
                if threshold > max_extent * 10:  # 10x data extent seems too large
                    return False
            
            return True
        except Exception:
            return True  # If we can't determine, assume it's reasonable
    
    def calculate_distance_matrix(
        self,
        origins: gpd.GeoDataFrame,
        destinations: gpd.GeoDataFrame,
        metric: str = "euclidean",
        network_graph: Optional[Any] = None
    ) -> np.ndarray:
        """
        Calculate distance matrix with caching for performance.
        
        Args:
            origins: GeoDataFrame of origin points
            destinations: GeoDataFrame of destination points
            metric: Distance metric ("euclidean", "manhattan", "network")
            network_graph: Optional network graph for network distances
            
        Returns:
            Distance matrix of shape (len(origins), len(destinations))
        
        Metrics:
        - euclidean: Straight-line distance
        - manhattan: Grid-based distance (L1 norm)
        - network: Road network distance (requires OSMnx)
        """
        # Check cache first
        cache_key = self._generate_cache_key(origins, destinations, metric)
        if cache_key in self._cache:
            logger.debug("Using cached distance matrix")
            return self._cache[cache_key]
        
        # Harmonize CRS where possible
        if origins.crs and destinations.crs and origins.crs != destinations.crs:
            destinations = destinations.to_crs(origins.crs)
        
        # Project to metric CRS if data is geographic or looks like lon/lat with missing CRS
        if (origins.crs and origins.crs.is_geographic) or (origins.crs is None and self._looks_like_lonlat(origins)):
            origins_proj = origins.to_crs("EPSG:3857") if origins.crs else origins.set_crs("EPSG:4326", inplace=False).to_crs("EPSG:3857")
            if destinations.crs:
                destinations_proj = destinations.to_crs("EPSG:3857")
            else:
                # Assume same as origins
                destinations_proj = destinations.set_crs("EPSG:4326", inplace=False).to_crs("EPSG:3857")
        else:
            origins_proj = origins
            destinations_proj = destinations
        
        # Calculate distance matrix
        if metric == "euclidean":
            result = self.euclidean_distance(origins_proj, destinations_proj)
        elif metric == "manhattan":
            result = self.manhattan_distance(origins_proj, destinations_proj)
        elif metric == "network":
            if network_graph is None:
                logger.warning("Network graph not provided, falling back to Euclidean distance")
                result = self.euclidean_distance(origins_proj, destinations_proj)
            else:
                result = self.network_distance(origins_proj, destinations_proj, network_graph)
        else:
            raise ValueError(f"Unknown distance metric: {metric}. Use 'euclidean', 'manhattan', or 'network'")
        
        # Cache the result
        self._manage_cache(cache_key, result)
        return result
    
    def euclidean_distance(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame
    ) -> np.ndarray:
        """Vectorized Euclidean distance calculation"""
        # Extract coordinates
        origin_coords = np.array([[geom.x, geom.y] for geom in origins.geometry])
        dest_coords = np.array([[geom.x, geom.y] for geom in destinations.geometry])
        
        # Calculate pairwise Euclidean distances
        distances = cdist(origin_coords, dest_coords, metric='euclidean')
        # Note: We keep exact zeros for coincident points to ensure proper coverage calculation
        
        return distances
    
    def manhattan_distance(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame
    ) -> np.ndarray:
        """Manhattan (L1) distance"""
        # Extract coordinates
        origin_coords = np.array([[geom.x, geom.y] for geom in origins.geometry])
        dest_coords = np.array([[geom.x, geom.y] for geom in destinations.geometry])
        
        # Calculate pairwise Manhattan distances
        distances = cdist(origin_coords, dest_coords, metric='cityblock')
        
        return distances
    
    def network_distance(
        self, 
        origins: gpd.GeoDataFrame, 
        destinations: gpd.GeoDataFrame, 
        network_graph
    ) -> np.ndarray:
        """
        Network-based distance using OSMnx.
        For future implementation.
        
        This would use shortest path algorithms on a road network graph.
        """
        try:
            import osmnx as ox
            import networkx as nx
            
            # This is a placeholder for future implementation
            # Would need to:
            # 1. Snap origin/destination points to nearest network nodes
            # 2. Calculate shortest paths between all pairs
            # 3. Return distance matrix
            
            logger.warning("Network distance calculation not fully implemented yet. Using Euclidean as fallback.")
            return self.euclidean_distance(origins, destinations)
            
        except ImportError:
            logger.error("OSMnx not installed. Install with: pip install osmnx")
            raise
    
    def calculate_coverage_matrix(
        self,
        origins: gpd.GeoDataFrame,
        destinations: gpd.GeoDataFrame,
        threshold: float,
        metric: str = "euclidean",
        user_unit_hint: str = None
    ) -> np.ndarray:
        """
        Calculate binary coverage matrix with user-friendly unit conversion.
        
        Args:
            threshold: Service radius threshold
            user_unit_hint: Optional hint from user about units ('km' or 'm')
        
        Returns:
            Binary matrix where 1 indicates destination is within threshold of origin
        """
        distances = self.calculate_distance_matrix(origins, destinations, metric)
        
        # Use smart unit conversion with user hint
        threshold_meters, unit_description = self._smart_unit_conversion(threshold, origins, user_unit_hint)
        
        # Log the conversion for debugging
        logger.info(f"Service radius conversion: {unit_description}")
        
        # Check if threshold seems reasonable
        if not self._detect_reasonable_threshold(threshold, origins):
            logger.warning(f"Service radius {threshold} seems unusually large for data extent. "
                          f"Please verify units. Converted to {threshold_meters:.0f} meters.")
        
        coverage = (distances <= threshold_meters).astype(int)
        return coverage
    
    def get_unit_info(self, threshold: float, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Get user-friendly information about unit conversion with suggestions.
        Returns a dictionary with conversion details and suggestions for users.
        """
        # Get suggestions for user confirmation
        suggestions = self._suggest_units_for_user(threshold, gdf)
        
        # Get current conversion (auto-detected)
        threshold_meters, unit_description = self._smart_unit_conversion(threshold, gdf)
        is_reasonable = self._detect_reasonable_threshold(threshold, gdf)
        
        return {
            "input_value": threshold,
            "converted_meters": threshold_meters,
            "unit_description": unit_description,
            "is_reasonable": is_reasonable,
            "needs_user_confirmation": True,  # Always ask user to confirm
            "suggestions": suggestions,
            "user_message": self._generate_user_message(threshold, suggestions)
        }
    
    def _generate_user_message(self, threshold: float, suggestions: Dict[str, Any]) -> str:
        """Generate a user-friendly message about unit conversion"""
        recommended = next((s for s in suggestions["suggestions"] if s["recommended"]), None)
        if recommended:
            return (f"Please confirm the units for your service radius of {threshold}. "
                   f"I recommend treating it as {recommended['unit']} (converts to {recommended['converted_meters']:.0f} meters) "
                   f"because {recommended['reason']}. "
                   f"Please confirm this is correct, or specify if you meant a different unit.")
        else:
            return f"Please specify the units for your service radius of {threshold} (kilometers or meters)."

