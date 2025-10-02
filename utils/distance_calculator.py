import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class DistanceCalculator:
    """Computes distance matrices using various metrics"""
    
    def calculate_distance_matrix(
        self,
        origins: gpd.GeoDataFrame,
        destinations: gpd.GeoDataFrame,
        metric: str = "euclidean",
        network_graph: Optional[Any] = None
    ) -> np.ndarray:
        """
        Calculate distance matrix.
        
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
        # Ensure both GeoDataFrames are in the same CRS
        if origins.crs != destinations.crs:
            destinations = destinations.to_crs(origins.crs)
        
        # For accurate distance calculations, project to a metric CRS if needed
        if origins.crs and origins.crs.is_geographic:
            # Project to Web Mercator for distance calculations
            origins_proj = origins.to_crs("EPSG:3857")
            destinations_proj = destinations.to_crs("EPSG:3857")
        else:
            origins_proj = origins
            destinations_proj = destinations
        
        if metric == "euclidean":
            return self.euclidean_distance(origins_proj, destinations_proj)
        elif metric == "manhattan":
            return self.manhattan_distance(origins_proj, destinations_proj)
        elif metric == "network":
            if network_graph is None:
                logger.warning("Network graph not provided, falling back to Euclidean distance")
                return self.euclidean_distance(origins_proj, destinations_proj)
            return self.network_distance(origins_proj, destinations_proj, network_graph)
        else:
            raise ValueError(f"Unknown distance metric: {metric}. Use 'euclidean', 'manhattan', or 'network'")
    
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
        
        # Convert threshold to meters if needed
        # If the original CRS is geographic (EPSG:4326), the threshold is likely in kilometers
        if origins.crs and origins.crs.is_geographic:
            # Convert kilometers to meters
            threshold_meters = threshold * 1000
        else:
            # Already in meters (projected CRS)
            threshold_meters = threshold
        
        coverage = (distances <= threshold_meters).astype(int)
        return coverage

