import json
import geopandas as gpd
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import zipfile
import tempfile
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
import logging

logger = logging.getLogger(__name__)

class ExportHandler:
    """Handles exporting solutions in various formats"""
    
    def export_solution_geojson(
        self,
        solution: Dict[str, Any],
        data: Dict[str, gpd.GeoDataFrame],
        filename: str = "solution.geojson"
    ) -> bytes:
        """Export solution as GeoJSON with facility locations and assignments"""
        try:
            # Create a copy of facilities data with solution info
            candidate_gdf = data.get('candidate_sites')
            if candidate_gdf is None:
                raise ValueError("No candidate sites data available")
            
            # Create solution GeoDataFrame with only selected facilities
            selected_indices = solution.get('selected_facilities', [])
            solution_gdf = candidate_gdf.iloc[selected_indices].copy()
            
            # Add solution metadata
            solution_gdf['selected'] = True
            solution_gdf['facility_id'] = selected_indices
            
            # Add assignment counts if available
            assignments = solution.get('assignments', {})
            assignment_counts = {}
            for demand_id, facility_id in assignments.items():
                assignment_counts[facility_id] = assignment_counts.get(facility_id, 0) + 1
            
            solution_gdf['num_assigned'] = [assignment_counts.get(idx, 0) for idx in selected_indices]
            
            # Convert to GeoJSON
            geojson_str = solution_gdf.to_json()
            return geojson_str.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error exporting GeoJSON: {e}")
            raise
    
    def export_solution_csv(
        self,
        solution: Dict[str, Any],
        filename: str = "solution.csv"
    ) -> bytes:
        """Export solution metrics and assignments as CSV"""
        try:
            # Create DataFrame with solution info
            data_rows = []
            
            # Add selected facilities
            for idx in solution.get('selected_facilities', []):
                data_rows.append({
                    'type': 'facility',
                    'index': idx,
                    'selected': True
                })
            
            # Add assignments
            for demand_id, facility_id in solution.get('assignments', {}).items():
                data_rows.append({
                    'type': 'assignment',
                    'demand_id': demand_id,
                    'facility_id': facility_id
                })
            
            df = pd.DataFrame(data_rows)
            
            # Add metrics as additional rows
            metrics_df = pd.DataFrame([{
                'type': 'metric',
                'metric_name': k,
                'value': v
            } for k, v in solution.get('metrics', {}).items()])
            
            # Combine
            full_df = pd.concat([df, metrics_df], ignore_index=True)
            
            # Convert to CSV
            csv_buffer = BytesIO()
            full_df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            raise
    
    def export_solution_shapefile(
        self,
        solution: Dict[str, Any],
        data: Dict[str, gpd.GeoDataFrame],
        filename: str = "solution.zip"
    ) -> bytes:
        """Export solution as Shapefile (zipped)"""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Create shapefile
                candidate_gdf = data.get('candidate_sites')
                if candidate_gdf is None:
                    raise ValueError("No candidate sites data available")
                
                selected_indices = solution.get('selected_facilities', [])
                solution_gdf = candidate_gdf.iloc[selected_indices].copy()
                solution_gdf['selected'] = True
                solution_gdf['facility_id'] = selected_indices
                
                # Save as shapefile
                shp_path = tmpdir_path / "solution.shp"
                solution_gdf.to_file(shp_path)
                
                # Zip all shapefile components
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in tmpdir_path.glob("solution.*"):
                        zipf.write(file, file.name)
                
                return zip_buffer.getvalue()
                
        except Exception as e:
            logger.error(f"Error exporting Shapefile: {e}")
            raise
    
    def generate_pdf_report(
        self,
        solution: Dict[str, Any],
        problem_metadata: Dict[str, Any],
        parameters: Dict[str, Any],
        filename: str = "report.pdf"
    ) -> bytes:
        """
        Generate comprehensive PDF report with:
        - Problem description and parameters
        - Solution summary and metrics
        - Academic references
        - Methodology description
        """
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#FF4B4B'),
                spaceAfter=30,
            )
            story.append(Paragraph("Spatial Optimization Solution Report", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Problem Information
            story.append(Paragraph("<b>Problem Type</b>", styles['Heading2']))
            story.append(Paragraph(problem_metadata.get('name', 'Unknown'), styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            story.append(Paragraph("<b>Description</b>", styles['Heading2']))
            story.append(Paragraph(problem_metadata.get('description', ''), styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Parameters
            story.append(Paragraph("<b>Parameters</b>", styles['Heading2']))
            param_data = [[k, str(v)] for k, v in parameters.items()]
            if param_data:
                param_table = Table([['Parameter', 'Value']] + param_data)
                param_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(param_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Solution Summary
            story.append(Paragraph("<b>Solution Summary</b>", styles['Heading2']))
            story.append(Paragraph(f"Status: {solution.get('status', 'Unknown')}", styles['Normal']))
            story.append(Paragraph(f"Objective Value: {solution.get('objective_value', 'N/A'):.2f}", styles['Normal']))
            story.append(Paragraph(f"Solution Time: {solution.get('solution_time', 0):.2f} seconds", styles['Normal']))
            story.append(Paragraph(f"Selected Facilities: {len(solution.get('selected_facilities', []))}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Metrics
            story.append(Paragraph("<b>Performance Metrics</b>", styles['Heading2']))
            metrics = solution.get('metrics', {})
            metric_data = [[k.replace('_', ' ').title(), f"{v:.2f}" if isinstance(v, (int, float)) else str(v)] 
                          for k, v in metrics.items()]
            if metric_data:
                metric_table = Table([['Metric', 'Value']] + metric_data)
                metric_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(metric_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Academic References
            story.append(Paragraph("<b>Academic References</b>", styles['Heading2']))
            refs = problem_metadata.get('academic_refs', [])
            for i, ref in enumerate(refs[:5], 1):  # Limit to first 5 references
                story.append(Paragraph(f"{i}. {ref}", styles['Normal']))
                story.append(Spacer(1, 0.05*inch))
            
            # Build PDF
            doc.build(story)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise
    
    def save_session(
        self,
        problem_state: Dict[str, Any],
        conversation_history: List[Dict],
        filename: str = "session.json"
    ) -> bytes:
        """
        Save complete session for reproducibility.
        Includes all parameters, data references, and solution.
        """
        try:
            session_data = {
                'problem_state': {
                    'problem_type': problem_state.get('problem_type'),
                    'parameters': problem_state.get('parameters', {}),
                    'constraints': problem_state.get('constraints', {}),
                    'solution': problem_state.get('solution')
                },
                'conversation_history': conversation_history,
                'data_summary': {
                    name: {
                        'num_features': len(gdf),
                        'columns': list(gdf.columns),
                        'crs': str(gdf.crs) if gdf.crs else None
                    }
                    for name, gdf in problem_state.get('data', {}).items()
                }
            }
            
            json_str = json.dumps(session_data, indent=2)
            return json_str.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            raise

