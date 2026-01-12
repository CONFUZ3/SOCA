"""
High-performance map visualization using pydeck (deck.gl/MapGL).
Much faster than Folium for large datasets due to WebGL rendering.
"""

import pydeck as pdk
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


class PyDeckVisualizer:
    """Creates high-performance WebGL maps using pydeck/deck.gl"""
    
    # Free basemap styles (no API key required)
    # Using Carto basemaps which are free and don't require tokens
    BASEMAP_STYLES = {
        'light': 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
        'dark': 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        'voyager': 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',
        'positron': 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
        'satellite': 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',  # Carto doesn't have satellite, use voyager
        # OpenStreetMap via Carto
        'osm': 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',
    }
    
    # Color constants (RGBA)
    COLORS = {
        'demand': [66, 133, 244, 180],      # Blue
        'candidate': [158, 158, 158, 150],   # Gray
        'generated': [255, 152, 0, 180],     # Orange
        'selected': [244, 67, 54, 255],      # Red
        'assignment': [158, 158, 158, 100],  # Light gray
        'violation': [244, 67, 54, 200],     # Red
        'service_area': [66, 133, 244, 50],  # Light blue
    }
    
    def __init__(self, basemap_style: str = 'light'):
        """
        Initialize PyDeck visualizer.
        
        Args:
            basemap_style: One of 'light', 'dark', 'voyager', 'positron', 'osm'
        """
        self.basemap_style = basemap_style
    
    def set_basemap(self, style: str):
        """Change the basemap style"""
        if style in self.BASEMAP_STYLES:
            self.basemap_style = style
        else:
            logger.warning(f"Unknown basemap style '{style}', using 'light'")
            self.basemap_style = 'light'
    
    def get_basemap_url(self) -> str:
        """Get the current basemap URL"""
        return self.BASEMAP_STYLES.get(self.basemap_style, self.BASEMAP_STYLES['light'])
    
    def create_map(
        self,
        data: Dict[str, gpd.GeoDataFrame],
        solution: Optional[Dict[str, Any]] = None,
        problem_type: Optional[str] = None,
        viz_config: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        basemap_style: Optional[str] = None,
        **kwargs  # Accept extra args for compatibility
    ) -> pdk.Deck:
        """
        Create high-performance WebGL map visualization.
        
        Args:
            data: Dictionary with 'demand_points' and/or 'candidate_sites' GeoDataFrames
            solution: Optimization solution dictionary
            problem_type: Type of optimization problem
            viz_config: Visualization configuration
            parameters: Problem parameters
            constraints: Problem constraints
            basemap_style: One of 'light', 'dark', 'voyager', 'positron', 'osm'
        
        Returns:
            pdk.Deck object for rendering with st.pydeck_chart()
        """
        try:
            layers = []
            
            demand_gdf = data.get('demand_points')
            candidate_gdf = data.get('candidate_sites')
            
            # Calculate view state from data bounds
            view_state = self._calculate_view_state(data)
            
            # Get selected facilities from solution
            selected_indices = solution.get('selected_facilities', []) if solution else []
            assignments = solution.get('assignments', {}) if solution else {}
            
            # 1. Add assignment lines (render first, below points)
            if solution and assignments and demand_gdf is not None and candidate_gdf is not None:
                assignment_layer = self._create_assignment_layer(
                    demand_gdf, candidate_gdf, assignments
                )
                if assignment_layer:
                    layers.append(assignment_layer)
            
            # 2. Add service areas (if enabled)
            if (solution and viz_config and viz_config.get('show_service_areas', False) 
                and candidate_gdf is not None and selected_indices):
                service_radius = solution.get('metrics', {}).get('service_radius')
                if service_radius:
                    # Get unit info from parameters if available
                    service_radius_unit = None
                    if parameters:
                        service_radius_unit = parameters.get('service_radius_unit')
                    service_layer = self._create_service_area_layer(
                        candidate_gdf, selected_indices, service_radius, service_radius_unit
                    )
                    if service_layer:
                        layers.append(service_layer)
            
            # 3. Add candidate sites
            if candidate_gdf is not None and len(candidate_gdf) > 0:
                candidate_layer = self._create_candidate_layer(
                    candidate_gdf, selected_indices
                )
                layers.append(candidate_layer)
            
            # 4. Add demand points
            if demand_gdf is not None and len(demand_gdf) > 0:
                demand_layer = self._create_demand_layer(demand_gdf, assignments)
                layers.append(demand_layer)
            
            # 5. Add selected facilities (on top)
            if candidate_gdf is not None and selected_indices:
                facility_layer = self._create_facility_layer(
                    candidate_gdf, selected_indices
                )
                layers.append(facility_layer)
            
            # Determine basemap style
            if basemap_style:
                map_style = self.BASEMAP_STYLES.get(basemap_style, self.get_basemap_url())
            else:
                map_style = self.get_basemap_url()
            
            # Create deck with free Carto basemap (no API key required)
            deck = pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                map_style=map_style,
                tooltip={
                    "html": "<b>{name}</b><br/>{info}",
                    "style": {
                        "backgroundColor": "white",
                        "color": "black",
                        "fontSize": "12px",
                        "padding": "8px"
                    }
                }
            )
            
            return deck
            
        except Exception as e:
            logger.error(f"Error creating pydeck map: {e}", exc_info=True)
            # Return empty deck on error with basemap
            return pdk.Deck(
                layers=[],
                initial_view_state=pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=10),
                map_style=self.get_basemap_url()
            )
    
    def _calculate_view_state(self, data: Dict[str, gpd.GeoDataFrame]) -> pdk.ViewState:
        """Calculate optimal view state from data bounds"""
        all_gdfs = [gdf for gdf in data.values() if gdf is not None and len(gdf) > 0]
        
        if not all_gdfs:
            return pdk.ViewState(latitude=40.7128, longitude=-74.0060, zoom=10)
        
        # Combine bounds
        bounds_list = [gdf.total_bounds for gdf in all_gdfs]
        minx = min(b[0] for b in bounds_list)
        miny = min(b[1] for b in bounds_list)
        maxx = max(b[2] for b in bounds_list)
        maxy = max(b[3] for b in bounds_list)
        
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2
        
        # Estimate zoom from extent
        lon_range = maxx - minx
        lat_range = maxy - miny
        max_range = max(lon_range, lat_range)
        
        if max_range > 10:
            zoom = 5
        elif max_range > 5:
            zoom = 7
        elif max_range > 1:
            zoom = 9
        elif max_range > 0.5:
            zoom = 10
        elif max_range > 0.1:
            zoom = 12
        else:
            zoom = 13
        
        return pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom,
            pitch=0,
            bearing=0
        )
    
    def _gdf_to_points_df(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Convert GeoDataFrame to DataFrame with lon/lat columns"""
        df = pd.DataFrame({
            'lon': [geom.x for geom in gdf.geometry],
            'lat': [geom.y for geom in gdf.geometry],
            'idx': range(len(gdf))
        })
        
        # Add other columns for tooltips
        for col in gdf.columns:
            if col != 'geometry':
                df[col] = gdf[col].values
        
        return df
    
    def _create_demand_layer(
        self, 
        gdf: gpd.GeoDataFrame,
        assignments: Dict[int, int]
    ) -> pdk.Layer:
        """Create demand points layer"""
        df = self._gdf_to_points_df(gdf)
        df['color'] = [self.COLORS['demand']] * len(df)
        df['name'] = [f"Demand Point {i}" for i in range(len(df))]
        
        # Add assignment info
        info_list = []
        for i in range(len(df)):
            info = ""
            if i in assignments:
                info = f"Assigned to Facility {assignments[i]}"
            info_list.append(info)
        df['info'] = info_list
        
        return pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position=['lon', 'lat'],
            get_color='color',
            get_radius=50,
            radius_min_pixels=4,
            radius_max_pixels=15,
            pickable=True,
            auto_highlight=True,
        )
    
    def _create_candidate_layer(
        self,
        gdf: gpd.GeoDataFrame,
        selected_indices: List[int]
    ) -> pdk.Layer:
        """Create candidate sites layer (excluding selected)"""
        df = self._gdf_to_points_df(gdf)
        
        # Color based on whether generated or not
        colors = []
        names = []
        for i in range(len(df)):
            if i in selected_indices:
                # Selected sites will be shown in facility layer
                colors.append([0, 0, 0, 0])  # Transparent
                names.append("")
            elif 'generated' in gdf.columns and gdf.iloc[i].get('generated', False):
                colors.append(self.COLORS['generated'])
                names.append(f"Generated Site {i}")
            else:
                colors.append(self.COLORS['candidate'])
                names.append(f"Candidate Site {i}")
        
        df['color'] = colors
        df['name'] = names
        df['info'] = [""] * len(df)
        
        return pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position=['lon', 'lat'],
            get_color='color',
            get_radius=60,
            radius_min_pixels=5,
            radius_max_pixels=18,
            pickable=True,
            auto_highlight=True,
        )
    
    def _create_facility_layer(
        self,
        gdf: gpd.GeoDataFrame,
        selected_indices: List[int]
    ) -> pdk.Layer:
        """Create selected facilities layer with prominent markers"""
        # Filter to only selected
        selected_rows = []
        for idx in selected_indices:
            if 0 <= idx < len(gdf):
                row = gdf.iloc[idx]
                is_generated = 'generated' in gdf.columns and row.get('generated', False)
                selected_rows.append({
                    'lon': row.geometry.x,
                    'lat': row.geometry.y,
                    'idx': idx,
                    'color': self.COLORS['generated'] if is_generated else self.COLORS['selected'],
                    'name': f"{'Generated ' if is_generated else ''}Facility {idx}",
                    'info': "SELECTED"
                })
        
        if not selected_rows:
            return None
        
        df = pd.DataFrame(selected_rows)
        
        # Use icon layer for facilities (star-like appearance)
        return pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position=['lon', 'lat'],
            get_color='color',
            get_radius=100,
            radius_min_pixels=10,
            radius_max_pixels=25,
            pickable=True,
            auto_highlight=True,
            stroked=True,
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2,
        )
    
    def _create_assignment_layer(
        self,
        demand_gdf: gpd.GeoDataFrame,
        facility_gdf: gpd.GeoDataFrame,
        assignments: Dict[int, int]
    ) -> pdk.Layer:
        """Create assignment lines layer"""
        lines = []
        
        for demand_idx, facility_idx in assignments.items():
            if (0 <= demand_idx < len(demand_gdf) and 
                0 <= facility_idx < len(facility_gdf)):
                d_geom = demand_gdf.iloc[demand_idx].geometry
                f_geom = facility_gdf.iloc[facility_idx].geometry
                
                lines.append({
                    'start': [d_geom.x, d_geom.y],
                    'end': [f_geom.x, f_geom.y],
                    'color': self.COLORS['assignment']
                })
        
        if not lines:
            return None
        
        df = pd.DataFrame(lines)
        
        return pdk.Layer(
            'LineLayer',
            data=df,
            get_source_position='start',
            get_target_position='end',
            get_color='color',
            get_width=1,
            width_min_pixels=1,
        )
    
    def _create_service_area_layer(
        self,
        facility_gdf: gpd.GeoDataFrame,
        selected_indices: List[int],
        radius: float,
        radius_unit: Optional[str] = None
    ) -> pdk.Layer:
        """Create service area circles
        
        Args:
            facility_gdf: GeoDataFrame with facility locations
            selected_indices: List of selected facility indices
            radius: Service radius value
            radius_unit: Unit of radius ('m', 'km', 'miles') - if None, will use smart conversion
        """
        circles = []
        
        # Convert radius to meters based on unit info
        radius_meters = self._convert_radius_to_meters(radius, radius_unit, facility_gdf)
        
        for idx in selected_indices:
            if 0 <= idx < len(facility_gdf):
                geom = facility_gdf.iloc[idx].geometry
                circles.append({
                    'lon': geom.x,
                    'lat': geom.y,
                    'radius': radius_meters,
                    'color': self.COLORS['service_area']
                })
        
        if not circles:
            return None
        
        df = pd.DataFrame(circles)
        
        return pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position=['lon', 'lat'],
            get_radius='radius',
            get_color='color',
            pickable=False,
            filled=True,
            stroked=False,
        )
    
    def _convert_radius_to_meters(
        self,
        radius: float,
        radius_unit: Optional[str],
        gdf: gpd.GeoDataFrame
    ) -> float:
        """Convert radius to meters using explicit unit specification.
        
        Args:
            radius: The radius value
            radius_unit: Unit ('m', 'km', 'miles') - if None, assumes meters with warning
            gdf: GeoDataFrame (unused, kept for API compatibility)
            
        Returns:
            Radius in meters
        """
        if radius_unit is None:
            logger.warning(
                f"PyDeck: No unit specified for radius {radius}. Assuming meters. "
                "Specify unit explicitly to avoid ambiguity."
            )
            return radius
        
        unit = radius_unit.lower().strip()
        if unit in ('km', 'kilometer', 'kilometers'):
            return radius * 1000
        elif unit in ('m', 'meter', 'meters'):
            return radius
        elif unit in ('mi', 'mile', 'miles'):
            return radius * 1609.34
        else:
            logger.warning(f"PyDeck: Unknown unit '{radius_unit}', assuming meters")
            return radius


# Singleton instance for caching
_pydeck_visualizer = None

def get_pydeck_visualizer() -> PyDeckVisualizer:
    """Get or create singleton PyDeckVisualizer instance"""
    global _pydeck_visualizer
    if _pydeck_visualizer is None:
        _pydeck_visualizer = PyDeckVisualizer()
    return _pydeck_visualizer

