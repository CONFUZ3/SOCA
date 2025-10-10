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
        try:
            xs = [geom.x for geom in gdf.geometry]
            ys = [geom.y for geom in gdf.geometry]
            if not xs or not ys:
                return False
            return (
                min(xs) >= -180 and max(xs) <= 180 and
                min(ys) >= -90 and max(ys) <= 90
            )
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
        metric: str = "euclidean"
    ) -> np.ndarray:
        """
        Calculate binary coverage matrix.
        
        Args:
            threshold: Service radius threshold. If origins are in geographic CRS (EPSG:4326),
                      this is assumed to be in kilometers and will be converted to meters.
        
        Returns:
            Binary matrix where 1 indicates destination is within threshold of origin
        """
        distances = self.calculate_distance_matrix(origins, destinations, metric)
        
        # Convert threshold to meters if needed (treat missing-CRS lon/lat as geographic)
        if (origins.crs and origins.crs.is_geographic) or (origins.crs is None and self._looks_like_lonlat(origins)):
            threshold_meters = threshold * 1000
        else:
            threshold_meters = threshold
        
        coverage = (distances <= threshold_meters).astype(int)
        return coverage

