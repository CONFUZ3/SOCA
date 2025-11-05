import folium
from folium import plugins
import branca.colormap as cm
import geopandas as gpd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import base64

logger = logging.getLogger(__name__)

class MapVisualizer:
    """Creates interactive Folium maps with solutions"""
    
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

    def create_map(
        self,
        data: Dict[str, gpd.GeoDataFrame],
        solution: Optional[Dict[str, Any]] = None,
        problem_type: Optional[str] = None,
        viz_config: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        raster_data: Optional[Dict[str, Any]] = None
    ) -> folium.Map:
        """
        Create comprehensive map visualization.
        
        Layers:
        - Base map (OpenStreetMap)
        - Raster overlay (optional satellite imagery)
        - Demand points (blue circles)
        - Candidate sites (gray circles)
        - Selected facilities (red stars)
        - Assignment lines (demand to facility)
        - Service areas (coverage circles/polygons)
        - Legend
        
        Args:
            raster_data: Optional dictionary of raster overlays to display as basemap
        """
        try:
            # Default visualization config
            default_config = {
                "facility_marker": {"color": "red", "size": 10, "icon": "star"},
                "demand_marker": {"color": "blue", "size": 5, "icon": "circle"},
                "candidate_marker": {"color": "gray", "size": 7, "icon": "circle"},
                "show_assignments": True,
                "assignment_line_color": "gray",
                "assignment_line_weight": 1,
                "assignment_line_opacity": 0.5,
                "show_service_areas": False
            }
            
            # Merge with provided config, using defaults for missing keys
            if viz_config is None:
                viz_config = default_config
            else:
                # Ensure all required keys are present
                for key, default_value in default_config.items():
                    if key not in viz_config:
                        viz_config[key] = default_value
            
            # Calculate map center
            all_gdfs = [gdf for gdf in data.values() if gdf is not None and len(gdf) > 0]
            if not all_gdfs:
                # Default center if no data
                center = [40.7128, -74.0060]  # New York City
                zoom = 12
            else:
                center = self._calculate_map_center(all_gdfs)
                zoom = self._calculate_zoom_level(all_gdfs)
        
            # Create base map
            m = folium.Map(
                location=center,
                zoom_start=zoom,
                tiles='OpenStreetMap'
            )
            
            # Add raster overlays first (so they appear as basemap)
            if raster_data:
                for raster_name, raster_info in raster_data.items():
                    try:
                        self._add_raster_overlay(m, raster_info, raster_name)
                    except Exception as e:
                        logger.warning(f"Could not add raster overlay {raster_name}: {e}")
            
            # Add data layers
            demand_gdf = data.get('demand_points')
            candidate_gdf = data.get('candidate_sites')
            
            # Add candidate sites first (so they're behind other markers)
            if candidate_gdf is not None and len(candidate_gdf) > 0:
                selected_indices = solution.get('selected_facilities', []) if solution else []
                self._add_candidate_sites(m, candidate_gdf, selected_indices, viz_config)
            
            # Add demand points
            if demand_gdf is not None and len(demand_gdf) > 0:
                assignments = solution.get('assignments', {}) if solution else None
                self._add_demand_points(m, demand_gdf, assignments, viz_config)
            
            # Add selected facilities (highlight them)
            if solution and candidate_gdf is not None:
                selected_indices = solution.get('selected_facilities', [])
                if selected_indices:
                    self._add_selected_facilities(m, candidate_gdf, selected_indices, viz_config)
            
            # Add assignment lines
            if (solution and viz_config.get('show_assignments', True) and 
                demand_gdf is not None and candidate_gdf is not None):
                assignments = solution.get('assignments', {})
                if assignments:
                    self._add_assignment_lines(m, demand_gdf, candidate_gdf, assignments, viz_config)
            
            # Add assignment violation indicators
            if (solution and demand_gdf is not None and candidate_gdf is not None):
                metrics = solution.get('metrics', {})
                violations = metrics.get('assignment_violations', [])
                if violations:
                    self._add_violation_indicators(m, demand_gdf, candidate_gdf, violations, viz_config)
            
            # Add service areas
            if (solution and viz_config.get('show_service_areas', False) and 
                candidate_gdf is not None):
                selected_indices = solution.get('selected_facilities', [])
                service_radius = solution.get('metrics', {}).get('service_radius')
                if service_radius:
                    # Get user unit hint from parameters if available
                    user_unit_hint = None
                    if parameters and 'user_unit_hint' in parameters:
                        user_unit_hint = parameters['user_unit_hint']
                    
                    self._add_service_areas(m, candidate_gdf, selected_indices, service_radius, viz_config, user_unit_hint)
            
            # Add legend
            self._add_legend(
                m,
                problem_type=problem_type,
                has_solution=solution is not None,
                parameters=parameters or {},
                constraints=constraints or {},
                solution=solution or {}
            )
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating map: {e}", exc_info=True)
            # Return a basic map with error message
            error_map = folium.Map(
                location=[40.7128, -74.0060],
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            folium.Marker(
                location=[40.7128, -74.0060],
                popup=f"Map Error: {str(e)}",
                icon=folium.Icon(color='red', icon='exclamation-triangle')
            ).add_to(error_map)
            return error_map
    
    def _add_demand_points(
        self, 
        map_obj: folium.Map, 
        gdf: gpd.GeoDataFrame, 
        assignments: Optional[Dict[int, int]],
        viz_config: Dict[str, Any]
    ):
        """Add demand points with optional color coding by assignment"""
        demand_layer = folium.FeatureGroup(name='Demand Points')
        
        # Use positional indexing to match solver indices
        for pos_idx in range(len(gdf)):
            row = gdf.iloc[pos_idx]
            # Get coordinates
            coords = [row.geometry.y, row.geometry.x]
            
            # Create popup text
            popup_text = f"<b>Demand Point {pos_idx}</b><br>"
            for col in gdf.columns:
                if col != 'geometry':
                    popup_text += f"{col}: {row[col]}<br>"
            
            if assignments and pos_idx in assignments:
                popup_text += f"<b>Assigned to: Facility {assignments[pos_idx]}</b>"
            
            # Add marker
            folium.CircleMarker(
                location=coords,
                radius=viz_config['demand_marker']['size'],
                color=viz_config['demand_marker']['color'],
                fill=True,
                fillColor=viz_config['demand_marker']['color'],
                fillOpacity=0.6,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"Demand {pos_idx}"
            ).add_to(demand_layer)
        
        demand_layer.add_to(map_obj)
    
    def _add_candidate_sites(
        self, 
        map_obj: folium.Map, 
        gdf: gpd.GeoDataFrame, 
        selected_indices: List[int],
        viz_config: Dict[str, Any]
    ):
        """Add candidate sites, highlighting selected ones"""
        candidate_layer = folium.FeatureGroup(name='Candidate Sites')
        
        # Use positional indexing to match solver indices
        for pos_idx in range(len(gdf)):
            row = gdf.iloc[pos_idx]
            coords = [row.geometry.y, row.geometry.x]
            
            # Determine if this site is selected (compare positional indices)
            is_selected = pos_idx in selected_indices
            
            # Create popup text
            is_generated = 'generated' in gdf.columns and row.get('generated', False)
            site_type = "Generated Candidate Site" if is_generated else "Candidate Site"
            popup_text = f"<b>{site_type} {pos_idx}</b><br>"
            if is_generated:
                popup_text += "<b style='color:orange;'>🔧 GENERATED</b><br>"
            if is_selected:
                popup_text += "<b style='color:red;'>✓ SELECTED</b><br>"
            for col in gdf.columns:
                if col != 'geometry' and col != 'generated':
                    popup_text += f"{col}: {row[col]}<br>"
            
            # Add marker with different styling for generated vs regular sites
            if is_generated:
                # Generated sites get a different color and style
                folium.CircleMarker(
                    location=coords,
                    radius=viz_config['candidate_marker']['size'] + 2,  # Slightly larger
                    color='orange',  # Orange color for generated sites
                    fill=True,
                    fillColor='orange',
                    fillOpacity=0.4 if not is_selected else 0.7,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"Generated Candidate {pos_idx}"
                ).add_to(candidate_layer)
            else:
                # Regular candidate sites
                folium.CircleMarker(
                    location=coords,
                    radius=viz_config['candidate_marker']['size'],
                    color=viz_config['candidate_marker']['color'],
                    fill=True,
                    fillColor=viz_config['candidate_marker']['color'],
                    fillOpacity=0.3 if not is_selected else 0.6,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"Candidate {pos_idx}"
                ).add_to(candidate_layer)
        
        candidate_layer.add_to(map_obj)
    
    def _add_selected_facilities(
        self, 
        map_obj: folium.Map, 
        gdf: gpd.GeoDataFrame, 
        selected_indices: List[int],
        viz_config: Dict[str, Any]
    ):
        """Add selected facilities with prominent markers"""
        facility_layer = folium.FeatureGroup(name='Selected Facilities')
        
        for pos_idx in selected_indices:
            # Use positional indexing - selected_indices are positional indices from solver
            if pos_idx < 0 or pos_idx >= len(gdf):
                logger.warning(f"Selected facility index {pos_idx} is out of bounds (gdf length: {len(gdf)})")
                continue
            
            row = gdf.iloc[pos_idx]
            coords = [row.geometry.y, row.geometry.x]
            
            # Check if this is a generated site
            is_generated = 'generated' in gdf.columns and row.get('generated', False)
            
            # Create popup text
            facility_type = "SELECTED GENERATED FACILITY" if is_generated else "SELECTED FACILITY"
            popup_text = f"<b>{facility_type} {pos_idx}</b><br>"
            if is_generated:
                popup_text += "<b style='color:orange;'>🔧 GENERATED</b><br>"
            popup_text += "<b style='color:red;'>✓ SELECTED</b><br>"
            for col in gdf.columns:
                if col != 'geometry' and col != 'generated':
                    popup_text += f"{col}: {row[col]}<br>"
            
            # Add prominent marker with different styling for generated vs regular
            if is_generated:
                # Generated selected facilities get a special icon
                folium.Marker(
                    location=coords,
                    icon=folium.Icon(
                        color='orange',
                        icon='star',
                        prefix='fa'
                    ),
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"Generated Facility {pos_idx}"
                ).add_to(facility_layer)
            else:
                # Regular selected facilities
                folium.Marker(
                    location=coords,
                    icon=folium.Icon(
                        color=viz_config['facility_marker']['color'],
                        icon='star',
                        prefix='fa'
                    ),
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"Facility {pos_idx}"
                ).add_to(facility_layer)
        
        facility_layer.add_to(map_obj)
    
    def _add_assignment_lines(
        self, 
        map_obj: folium.Map, 
        demand_gdf: gpd.GeoDataFrame, 
        facility_gdf: gpd.GeoDataFrame,
        assignments: Dict[int, int],
        viz_config: Dict[str, Any]
    ):
        """Draw lines connecting demand points to assigned facilities"""
        assignment_layer = folium.FeatureGroup(name='Assignments')
        
        # assignments contains positional indices from solver
        for demand_idx, facility_idx in assignments.items():
            # Validate indices are within bounds
            if demand_idx < 0 or demand_idx >= len(demand_gdf):
                logger.warning(f"Assignment demand index {demand_idx} is out of bounds (demand gdf length: {len(demand_gdf)})")
                continue
            if facility_idx < 0 or facility_idx >= len(facility_gdf):
                logger.warning(f"Assignment facility index {facility_idx} is out of bounds (facility gdf length: {len(facility_gdf)})")
                continue
            
            demand_point = demand_gdf.iloc[demand_idx].geometry
            facility_point = facility_gdf.iloc[facility_idx].geometry
            
            coords = [
                [demand_point.y, demand_point.x],
                [facility_point.y, facility_point.x]
            ]
            
            folium.PolyLine(
                coords,
                color=viz_config['assignment_line_color'],
                weight=viz_config['assignment_line_weight'],
                opacity=viz_config['assignment_line_opacity']
            ).add_to(assignment_layer)
        
        assignment_layer.add_to(map_obj)
    
    def _add_violation_indicators(
        self, 
        map_obj: folium.Map, 
        demand_gdf: gpd.GeoDataFrame, 
        facility_gdf: gpd.GeoDataFrame,
        violations: List[Dict[str, Any]],
        viz_config: Dict[str, Any]
    ):
        """Add visual indicators for assignment violations (assignments exceeding service radius)"""
        violation_layer = folium.FeatureGroup(name='Assignment Violations')
        
        for violation in violations:
            demand_idx = violation['demand_idx']
            facility_idx = violation['facility_idx']
            distance = violation['distance']
            threshold = violation['threshold']
            excess = violation['excess']
            
            # Validate indices
            if (demand_idx < 0 or demand_idx >= len(demand_gdf) or 
                facility_idx < 0 or facility_idx >= len(facility_gdf)):
                continue
            
            demand_point = demand_gdf.iloc[demand_idx].geometry
            facility_point = facility_gdf.iloc[facility_idx].geometry
            
            # Create violation line with red color and thicker weight
            coords = [
                [demand_point.y, demand_point.x],
                [facility_point.y, facility_point.x]
            ]
            
            folium.PolyLine(
                coords,
                color='red',
                weight=3,
                opacity=0.8,
                popup=f"VIOLATION: Distance {distance:.2f} > Threshold {threshold:.2f} (Excess: {excess:.2f})"
            ).add_to(violation_layer)
            
            # Add violation marker at demand point
            folium.CircleMarker(
                location=[demand_point.y, demand_point.x],
                radius=8,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.7,
                popup=f"VIOLATION: Demand {demand_idx} assigned to Facility {facility_idx}<br>"
                      f"Distance: {distance:.2f} > Threshold: {threshold:.2f}<br>"
                      f"Excess: {excess:.2f}"
            ).add_to(violation_layer)
        
        violation_layer.add_to(map_obj)
    
    def _add_service_areas(
        self, 
        map_obj: folium.Map, 
        facility_gdf: gpd.GeoDataFrame, 
        selected_indices: List[int],
        radius: float,
        viz_config: Dict[str, Any],
        user_unit_hint: str = None
    ):
        """Add service area circles around facilities with user-friendly unit conversion"""
        service_layer = folium.FeatureGroup(name='Service Areas')
        
        # Use the same smart unit conversion as distance calculator
        from utils.distance_calculator import DistanceCalculator
        dist_calc = DistanceCalculator()
        radius_meters, unit_description = dist_calc._smart_unit_conversion(radius, facility_gdf, user_unit_hint)
        
        # Log the conversion for consistency
        logger.info(f"Visualization radius conversion: {unit_description}")
        
        for pos_idx in selected_indices:
            # Use positional indexing - selected_indices are positional indices from solver
            if pos_idx < 0 or pos_idx >= len(facility_gdf):
                logger.warning(f"Service area index {pos_idx} is out of bounds (gdf length: {len(facility_gdf)})")
                continue
            
            row = facility_gdf.iloc[pos_idx]
            coords = [row.geometry.y, row.geometry.x]
            
            folium.Circle(
                location=coords,
                radius=radius_meters,
                color=viz_config.get('service_area_color', 'blue'),
                fill=True,
                fillOpacity=viz_config.get('service_area_opacity', 0.2),
                popup=f"Service Area: Facility {pos_idx}"
            ).add_to(service_layer)
        
        service_layer.add_to(map_obj)
    
    def _add_legend(
        self,
        map_obj: folium.Map,
        problem_type: Optional[str],
        has_solution: bool,
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        solution: Dict[str, Any]
    ):
        """Add dynamic legend with problem-specific info, parameters, constraints, and key metrics"""
        # Extract common metrics if present
        metrics = solution.get('metrics', {}) if has_solution else {}
        objective_value = solution.get('objective_value') if has_solution else None

        # Build dynamic problem-specific info
        problem_info_html = ''
        if problem_type:
            pt = (problem_type or '').lower()
            if pt == 'p-median':
                objective = parameters.get('objective', 'total')
                avg_dist = metrics.get('average_distance')
                total_dist = metrics.get('total_weighted_distance')
                problem_info_html += f"<p><b>Objective:</b> Minimize {objective} distance</p>"
                if objective_value is not None:
                    problem_info_html += f"<p><b>Objective Value:</b> {objective_value:.2f}</p>"
                if avg_dist is not None:
                    problem_info_html += f"<p><b>Avg Distance:</b> {avg_dist:.2f}</p>"
                if total_dist is not None:
                    problem_info_html += f"<p><b>Total Distance:</b> {total_dist:.2f}</p>"
            elif pt == 'p-center':
                max_dist = metrics.get('max_distance')
                problem_info_html += "<p><b>Objective:</b> Minimize maximum distance</p>"
                if objective_value is not None:
                    problem_info_html += f"<p><b>W (max dist):</b> {objective_value:.2f}</p>"
                if max_dist is not None:
                    problem_info_html += f"<p><b>Max Distance:</b> {max_dist:.2f}</p>"
            elif pt == 'mclp':
                sr = parameters.get('service_radius') or metrics.get('service_radius')
                cov = metrics.get('coverage_percentage')
                problem_info_html += "<p><b>Objective:</b> Maximize coverage</p>"
                if sr is not None:
                    problem_info_html += f"<p><b>Service Radius:</b> {sr}</p>"
                if cov is not None:
                    problem_info_html += f"<p><b>Coverage:</b> {cov:.1f}%</p>"
            elif pt == 'lscp':
                sr = parameters.get('service_radius') or metrics.get('service_radius')
                nfac = metrics.get('num_facilities') or objective_value
                problem_info_html += "<p><b>Objective:</b> Minimize facilities for full coverage</p>"
                if sr is not None:
                    problem_info_html += f"<p><b>Service Radius:</b> {sr}</p>"
                if nfac is not None:
                    try:
                        nfac_str = f"{int(nfac)}"
                    except Exception:
                        nfac_str = f"{nfac}"
                    problem_info_html += f"<p><b>Facilities:</b> {nfac_str}</p>"
            else:
                problem_info_html += f"<p><b>Problem:</b> {problem_type}</p>"

        # Constraint summary (compact)
        constraint_bits = []
        if constraints:
            must_inc = constraints.get('must_include')
            must_exc = constraints.get('must_exclude')
            max_fac = constraints.get('max_facilities')
            distance_cap = constraints.get('max_distance') or constraints.get('distance_threshold')
            if must_inc:
                constraint_bits.append(f"Must include: {len(must_inc)}")
            if must_exc:
                constraint_bits.append(f"Must exclude: {len(must_exc)}")
            if max_fac is not None:
                constraint_bits.append(f"Max facilities: {max_fac}")
            if distance_cap is not None:
                constraint_bits.append(f"Max distance: {distance_cap}")

        constraints_html = ''
        if constraint_bits:
            constraints_html = "<p><b>Constraints:</b> " + ", ".join(constraint_bits) + "</p>"

        # Core legend entries
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 260px; height: auto; 
                    background-color: white; z-index:9999; font-size:13px;
                    border:2px solid grey; border-radius: 5px; padding: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.2)">
        <h4 style="margin-top:0; margin-bottom:6px;">Legend</h4>
        '''

        if problem_type:
            legend_html += f'<p style="margin:0 0 6px 0;"><b>Problem:</b> {problem_type}</p>'

        legend_html += '''
        <p style="margin:0;"><span style="color:blue;">●</span> Demand Points</p>
        <p style="margin:0;"><span style="color:gray;">●</span> Candidate Sites</p>
        <p style="margin:0;"><span style="color:orange;">●</span> Generated Sites</p>
        '''

        if has_solution:
            legend_html += '<p style="margin:0;"><span style="color:red;">★</span> Selected Facilities</p>'
            legend_html += '<p style="margin:0;"><span style="color:orange;">★</span> Selected Generated</p>'
            legend_html += '<p style="margin:0;"><span style="color:gray;">─</span> Assignments</p>'
            
            # Check for violations in metrics
            violations = solution.get('metrics', {}).get('assignment_violations', [])
            if violations:
                legend_html += '<p style="margin:0;"><span style="color:red;">●</span> Assignment Violations</p>'
                legend_html += '<p style="margin:0;"><span style="color:red;">─</span> Violation Lines</p>'

        # Add problem-specific and constraints info
        if problem_info_html:
            legend_html += f'<div style="margin-top:6px;">{problem_info_html}</div>'
        if constraints_html:
            legend_html += f'<div style="margin-top:4px;">{constraints_html}</div>'

        legend_html += '</div>'

        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def _calculate_map_center(self, gdfs: List[gpd.GeoDataFrame]) -> List[float]:
        """Calculate center point of all data for map initialization"""
        # Combine all bounds
        all_bounds = []
        for gdf in gdfs:
            bounds = gdf.total_bounds  # minx, miny, maxx, maxy
            all_bounds.append(bounds)
        
        # Calculate overall bounds
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        
        # Calculate center
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2
        
        return [center_lat, center_lon]
    
    def _calculate_zoom_level(self, gdfs: List[gpd.GeoDataFrame]) -> int:
        """Estimate appropriate zoom level based on data extent"""
        # Combine all bounds
        all_bounds = []
        for gdf in gdfs:
            bounds = gdf.total_bounds
            all_bounds.append(bounds)
        
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        
        # Calculate extent in degrees
        lon_range = maxx - minx
        lat_range = maxy - miny
        max_range = max(lon_range, lat_range)
        
        # Estimate zoom level (rough heuristic)
        if max_range > 10:
            return 6
        elif max_range > 5:
            return 8
        elif max_range > 1:
            return 10
        elif max_range > 0.5:
            return 11
        elif max_range > 0.1:
            return 12
        else:
            return 13
    
    def _add_raster_overlay(
        self,
        map_obj: folium.Map,
        raster_info: Dict[str, Any],
        raster_name: str
    ):
        """
        Add raster overlay to map using Folium ImageOverlay.
        
        Args:
            map_obj: Folium map object
            raster_info: Dictionary with 'bounds', 'image_bytes', and other metadata
            raster_name: Name of the raster file
        """
        try:
            # Get bounds in format [[south, west], [north, east]]
            bounds = raster_info['bounds']
            
            # Convert image bytes to base64 data URI for Folium
            image_bytes = raster_info['image_bytes']
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_data_uri = f"data:image/png;base64,{image_base64}"
            
            # Create ImageOverlay
            # Note: Folium ImageOverlay expects bounds as [[south, west], [north, east]]
            overlay = folium.raster_layers.ImageOverlay(
                image=image_data_uri,
                bounds=bounds,
                name=raster_name,
                opacity=0.7,  # Slightly transparent so basemap shows through if needed
                interactive=True,
                cross_origin=False,
                zindex=1  # Below vector data but above basemap
            )
            
            overlay.add_to(map_obj)
            
            logger.info(f"Added raster overlay {raster_name} with bounds {bounds}")
            
        except Exception as e:
            logger.error(f"Error adding raster overlay {raster_name}: {e}", exc_info=True)
            raise

