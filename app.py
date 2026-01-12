import streamlit as st
import folium
from streamlit_folium import st_folium
import pydeck as pdk
import geopandas as gpd
from pathlib import Path
import os
import logging
import time
import inspect
import hashlib
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('spopt_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Imports from our modules
from agent.conversation_manager import ConversationManager
from solvers.registry import problem_registry
from utils.data_processor import DataProcessor
from utils.visualizer import MapVisualizer
from utils.pydeck_visualizer import PyDeckVisualizer
from utils.export_handler import ExportHandler
from config.settings import settings


# ============================================================================
# PERFORMANCE: Caching functions for expensive operations
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_data_hash(data_dict: dict) -> str:
    """Generate hash of data for cache invalidation"""
    hash_parts = []
    for name, gdf in data_dict.items():
        if gdf is not None:
            hash_parts.append(f"{name}:{len(gdf)}:{gdf.total_bounds.tobytes().hex()}")
    return hashlib.md5(":".join(hash_parts).encode()).hexdigest()


@st.cache_data(ttl=300, show_spinner=False)
def get_solution_hash(solution: dict) -> str:
    """Generate hash of solution for cache invalidation"""
    if not solution:
        return "none"
    key_parts = [
        str(solution.get('status', '')),
        str(solution.get('objective_value', '')),
        str(sorted(solution.get('selected_facilities', []))),
    ]
    return hashlib.md5(":".join(key_parts).encode()).hexdigest()

# Page config
st.set_page_config(
    page_title="Spatial Optimization Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": """Welcome to the Spatial Optimization Conversational Agent!

I'm here to help you solve facility location problems using state-of-the-art optimization techniques.

**To get started:**
1. Upload your geospatial data using the sidebar
   - **Demand points** (required): Locations with population/demand
   - **Candidate sites** (optional): Potential facility locations
2. Describe your optimization problem in natural language
3. I'll guide you through the process and help you find the optimal solution

**New Feature: Automatic Candidate Site Generation**
If you only upload demand data (no candidate sites), I'll automatically generate 100 random candidate sites within your demand extent. You can adjust the count and set a random seed for reproducibility in the sidebar.

**I can help you with:**
- P-Median: Minimize average/total distance
- P-Center: Minimize maximum distance (worst-case)
- MCLP: Maximize coverage within a service radius
- LSCP: Minimize facilities needed for full coverage

What problem would you like to solve today?"""
        })
    
    if "problem_state" not in st.session_state:
        st.session_state.problem_state = {
            "problem_type": None,
            "parameters": {},
            "constraints": {},
            "data": {},
            "solution": None,
            "solution_history": []
        }
    
    if "raster_data" not in st.session_state:
        st.session_state.raster_data = {}  # Store raster overlays separately from vector data
    
    if "map_renderer" not in st.session_state:
        st.session_state.map_renderer = "pydeck"  # Default to faster pydeck renderer
    
    if "basemap_style" not in st.session_state:
        st.session_state.basemap_style = "light"  # Default basemap style
    
    if "pydeck_visualizer" not in st.session_state:
        st.session_state.pydeck_visualizer = PyDeckVisualizer(basemap_style="light")
    
    if "conversation_manager" not in st.session_state:
        # Get API key
        api_key = None
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            st.error("GEMINI_API_KEY not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
            st.stop()
        
        st.session_state.conversation_manager = ConversationManager(
            api_key=api_key,
            problem_registry=problem_registry
        )
    
    if "data_processor" not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    
    if "map_visualizer" not in st.session_state:
        st.session_state.map_visualizer = MapVisualizer()
    
    if "export_handler" not in st.session_state:
        st.session_state.export_handler = ExportHandler()

initialize_session_state()

# Sidebar
with st.sidebar:
    st.title("Spatial Optimization")
    
    st.divider()
    
    # File upload section
    st.subheader("Upload Data")
    st.markdown("Upload geospatial data files (GeoJSON, Shapefile, CSV)")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["geojson", "json", "csv", "shp", "zip"],
        accept_multiple_files=True,
        help="Upload demand points, candidate sites, or boundary files",
        key="file_uploader"
    )
    
    if uploaded_files:
        new_files_loaded = False
        for file in uploaded_files:
            if file.name not in st.session_state.problem_state["data"]:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        gdf = st.session_state.data_processor.load_file(file)
                        gdf = st.session_state.data_processor.preprocess_data(gdf)
                        st.session_state.problem_state["data"][file.name] = gdf
                        st.success(f"✓ Loaded {file.name}: {len(gdf)} features")
                        new_files_loaded = True
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {str(e)}")

        # Immediately notify the model about uploaded data and skip confirmations
        if new_files_loaded and st.session_state.problem_state["data"]:
            data_summary = {}
            for name, gdf in st.session_state.problem_state["data"].items():
                try:
                    dtypes = {c: str(gdf[c].dtype) for c in gdf.columns if c != 'geometry'}
                except Exception:
                    dtypes = {}
                
                # Detect special columns
                data_processor = st.session_state.data_processor
                capacity_cols = data_processor.identify_capacity_columns(gdf)
                cost_cols = data_processor.identify_cost_columns(gdf)
                demand_cols = data_processor.identify_demand_columns(gdf)
                
                data_summary[name] = {
                    "num_features": len(gdf),
                    "geometry_type": gdf.geometry.type.unique()[0] if len(gdf) > 0 else "Unknown",
                    "columns": [c for c in gdf.columns if c != 'geometry'],
                    "dtypes": dtypes,
                    "bounds": gdf.total_bounds.tolist() if len(gdf) > 0 else [],
                    "capacity_columns": capacity_cols,
                    "cost_columns": cost_cols,
                    "demand_columns": demand_cols
                }
            try:
                with st.spinner("Syncing uploaded data with AI..."):
                    result = st.session_state.conversation_manager.notify_data_uploaded(
                        conversation_history=st.session_state.messages,
                        problem_state=st.session_state.problem_state,
                        uploaded_data_summary=data_summary
                    )
                # Update state and chat history silently
                st.session_state.problem_state = result["updated_state"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"]
                })
            except Exception as e:
                st.warning(f"Could not sync uploaded data to AI: {e}")
    
    # Raster/satellite imagery upload section
    st.divider()
    st.subheader("Upload Satellite Imagery")
    st.markdown("Upload raster files (GeoTIFF) to use as basemap overlay")
    
    uploaded_raster = st.file_uploader(
        "Choose raster file",
        type=["tif", "tiff", "geotiff"],
        accept_multiple_files=False,
        help="Upload a GeoTIFF raster file to use as basemap overlay",
        key="raster_uploader"
    )
    
    if uploaded_raster:
        if uploaded_raster.name not in st.session_state.raster_data:
            with st.spinner(f"Processing raster {uploaded_raster.name}..."):
                try:
                    raster_info = st.session_state.data_processor.load_raster_file(uploaded_raster)
                    st.session_state.raster_data[uploaded_raster.name] = raster_info
                    st.success(f"✓ Loaded raster {uploaded_raster.name}: {raster_info['width']}x{raster_info['height']} pixels")
                except Exception as e:
                    st.error(f"Error loading raster {uploaded_raster.name}: {str(e)}")
                    logger.error(f"Raster loading error: {e}", exc_info=True)
        else:
            st.info(f"Raster {uploaded_raster.name} already loaded")
    
    # Display loaded raster info
    if st.session_state.raster_data:
        st.divider()
        st.subheader("Loaded Raster")
        for name, raster_info in st.session_state.raster_data.items():
            with st.expander(f"{name}"):
                st.write(f"**Size:** {raster_info['width']}x{raster_info['height']} pixels")
                st.write(f"**CRS:** {raster_info['crs']}")
                bounds = raster_info['bounds']
                st.write(f"**Bounds:** [{bounds[0][0]:.6f}, {bounds[0][1]:.6f}] to [{bounds[1][0]:.6f}, {bounds[1][1]:.6f}]")
                if st.button(f"Remove {name}", key=f"remove_raster_{name}"):
                    del st.session_state.raster_data[name]
                    st.rerun()
    
    # Display loaded data summary
    if st.session_state.problem_state["data"]:
        st.divider()
        st.subheader("Loaded Data")
        for name, gdf in st.session_state.problem_state["data"].items():
            with st.expander(f"{name}"):
                st.write(f"**Features:** {len(gdf)}")
                st.write(f"**Geometry:** {gdf.geometry.type.unique()[0]}")
                st.write(f"**Columns:** {', '.join([c for c in gdf.columns if c != 'geometry'])}")
                st.write(f"**CRS:** {gdf.crs}")
        
        # Check if we have demand data but no candidate sites
        has_demand = False
        has_candidates = False
        
        for name, gdf in st.session_state.problem_state["data"].items():
            data_type = st.session_state.data_processor.identify_data_type(gdf)
            if data_type == "demand_points" or "demand" in name.lower():
                has_demand = True
            elif data_type == "candidate_sites" or any(word in name.lower() for word in ['candidate', 'site', 'facility']):
                has_candidates = True
        
        # Show candidate generation controls if we have demand but no candidates
        if has_demand and not has_candidates:
            st.divider()
            st.subheader("Candidate Site Generation")
            st.info("No candidate sites detected - will generate random sites within demand extent")
            
            # Initialize session state for generation controls
            if "generated_sites_count" not in st.session_state:
                st.session_state.generated_sites_count = 100
            if "generated_sites_seed" not in st.session_state:
                st.session_state.generated_sites_seed = None
            
            # Number of sites control
            st.session_state.generated_sites_count = st.number_input(
                "Number of candidate sites",
                min_value=10,
                max_value=500,
                value=st.session_state.generated_sites_count,
                help="Number of random candidate sites to generate within demand extent"
            )
            
            # Random seed control
            seed_input = st.text_input(
                "Random seed (optional)",
                value=str(st.session_state.generated_sites_seed) if st.session_state.generated_sites_seed is not None else "",
                help="Enter a number for reproducible results, or leave empty for random generation"
            )
            
            # Parse seed input
            if seed_input.strip():
                try:
                    st.session_state.generated_sites_seed = int(seed_input.strip())
                except ValueError:
                    st.warning("Invalid seed value. Please enter a number or leave empty.")
                    st.session_state.generated_sites_seed = None
            else:
                st.session_state.generated_sites_seed = None
    
    st.divider()
    
    # Removed Available Problems section from UI
    
    # Current problem state
    if st.session_state.problem_state["problem_type"]:
        st.subheader("Current Problem")
        st.info(st.session_state.problem_state["problem_type"])
        
        if st.session_state.problem_state["parameters"]:
            with st.expander("Parameters"):
                st.json(st.session_state.problem_state["parameters"])
        
        if st.session_state.problem_state["constraints"]:
            with st.expander("Constraints"):
                st.json(st.session_state.problem_state["constraints"])
    
    # Map renderer settings
    st.divider()
    st.subheader("⚡ Performance")
    
    renderer_options = {
        "pydeck": "PyDeck (Fast - WebGL)",
        "folium": "Folium (Classic - Leaflet)"
    }
    
    st.session_state.map_renderer = st.radio(
        "Map Renderer",
        options=list(renderer_options.keys()),
        format_func=lambda x: renderer_options[x],
        index=0 if st.session_state.get("map_renderer", "pydeck") == "pydeck" else 1,
        help="PyDeck uses WebGL for faster rendering with large datasets"
    )
    
    # Basemap selector (only for PyDeck)
    if st.session_state.get("map_renderer", "pydeck") == "pydeck":
        basemap_options = {
            "light": "🌤️ Light (Positron)",
            "dark": "🌙 Dark Matter",
            "voyager": "🗺️ Voyager (Colorful)",
        }
        
        if "basemap_style" not in st.session_state:
            st.session_state.basemap_style = "light"
        
        st.session_state.basemap_style = st.selectbox(
            "Basemap Style",
            options=list(basemap_options.keys()),
            format_func=lambda x: basemap_options[x],
            index=list(basemap_options.keys()).index(st.session_state.get("basemap_style", "light")),
            help="Choose the map background style"
        )
    
    # Clear conversation button
    st.divider()
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.problem_state = {
            "problem_type": None,
            "parameters": {},
            "constraints": {},
            "data": st.session_state.problem_state.get("data", {}),  # Keep data
            "solution": None,
            "solution_history": []
        }
        st.rerun()

# Main content area
st.title("Spatial Optimization Conversational Agent")

# Create two columns: chat and map
col1, col2 = st.columns([1, 1])


# ============================================================================
# METRICS DISPLAY: Problem-specific metrics formatting
# ============================================================================

def _display_problem_metrics(problem_type: str, solution: dict, metrics: dict):
    """Display problem-specific metrics with proper grouping and formatting."""
    
    # Common header metrics row
    metric_cols = st.columns(3)
    
    with metric_cols[0]:
        # Show objective value with problem-specific label
        obj_name = metrics.get('objective_name', 'objective_value')
        obj_val = metrics.get('objective_value', solution.get('objective_value'))
        if obj_val is not None:
            label = _format_metric_label(obj_name)
            st.metric(label, f"{obj_val:.2f}")
    
    with metric_cols[1]:
        st.metric("Solution Status", solution.get('status', 'Unknown').title())
    
    with metric_cols[2]:
        st.metric("Solution Time", f"{solution.get('solution_time', 0):.2f}s")
    
    # Problem-specific key metrics
    if problem_type == "lscp":
        _display_lscp_metrics(metrics)
    elif problem_type == "mclp":
        _display_mclp_metrics(metrics, solution)
    elif problem_type == "p-center":
        _display_pcenter_metrics(metrics)
    elif problem_type == "p-median":
        _display_pmedian_metrics(metrics)
    else:
        # Generic fallback
        _display_generic_metrics(metrics)


def _format_metric_label(key: str) -> str:
    """Format metric key as readable label."""
    labels = {
        'min_facilities': 'Min Facilities',
        'max_distance': 'Max Distance',
        'total_weighted_distance': 'Total Weighted Distance',
        'average_weighted_distance': 'Avg Weighted Distance',
        'covered_demand': 'Covered Demand',
        'served_demand': 'Served Demand',
        'expected_covered_demand': 'Expected Coverage',
    }
    return labels.get(key, key.replace('_', ' ').title())


def _display_lscp_metrics(metrics: dict):
    """Display LSCP-specific metrics."""
    st.markdown("##### Coverage Metrics")
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Facilities Required", int(metrics.get('num_facilities', 0)))
    with cols[1]:
        coverage = metrics.get('coverage_percentage', 0)
        st.metric("Coverage", f"{coverage:.1f}%")
    with cols[2]:
        radius = metrics.get('service_radius', 0)
        st.metric("Service Radius", f"{radius:.2f}")
    
    # Additional details
    with st.expander("Detailed Metrics"):
        st.write(f"**Total Demand Points:** {metrics.get('total_demand_points', 0)}")
        st.write(f"**Covered Points:** {metrics.get('num_covered_points', 0)}")
        st.write(f"**Uncovered Points:** {metrics.get('num_uncovered_points', 0)}")
        st.write(f"**Average Distance:** {metrics.get('average_distance', 0):.2f}")
        st.write(f"**Max Distance:** {metrics.get('max_distance', 0):.2f}")


def _display_mclp_metrics(metrics: dict, solution: dict):
    """Display MCLP-specific metrics."""
    variant = solution.get('variant_used', 'classical')
    
    st.markdown("##### Coverage Metrics")
    cols = st.columns(4)
    
    with cols[0]:
        coverage = metrics.get('coverage_percentage', 0)
        st.metric("Coverage", f"{coverage:.1f}%")
    with cols[1]:
        covered = metrics.get('covered_demand', 0)
        st.metric("Covered Demand", f"{covered:.1f}")
    with cols[2]:
        uncovered = metrics.get('uncovered_demand', 0)
        st.metric("Uncovered Demand", f"{uncovered:.1f}")
    with cols[3]:
        radius = metrics.get('service_radius', 0)
        st.metric("Service Radius", f"{radius:.2f}")
    
    # Variant-specific metrics
    with st.expander("Detailed Metrics"):
        st.write(f"**Variant:** {variant.title()}")
        st.write(f"**Facilities Selected:** {metrics.get('num_facilities', 0)}")
        st.write(f"**Total Demand:** {metrics.get('total_demand', 0):.1f}")
        st.write(f"**Covered Points:** {metrics.get('num_covered_points', 0)}")
        st.write(f"**Avg Distance (covered):** {metrics.get('average_distance_covered', 0):.2f}")
        
        # Variant-specific details
        if variant == 'capacitated':
            st.write(f"**Capacity Utilization:** {metrics.get('capacity_utilization', 0)*100:.1f}%")
            if 'avg_facility_utilization' in metrics:
                st.write(f"**Avg Facility Utilization:** {metrics.get('avg_facility_utilization', 0)*100:.1f}%")
        elif variant == 'budget':
            st.write(f"**Total Cost:** {metrics.get('total_cost', 0):.2f}")
            st.write(f"**Budget Utilization:** {metrics.get('budget_utilization', 0)*100:.1f}%")
        elif variant in ('multi_coverage', 'backup'):
            st.write(f"**K Required:** {metrics.get('k_required', 2)}")
            st.write(f"**Min Coverage Count:** {metrics.get('min_coverage_count', 0)}")
        elif variant == 'probabilistic':
            st.write(f"**Avg Reliability:** {metrics.get('avg_selected_reliability', 0)*100:.1f}%")


def _display_pcenter_metrics(metrics: dict):
    """Display P-Center-specific metrics."""
    st.markdown("##### Distance Metrics (Minimax)")
    cols = st.columns(4)
    
    with cols[0]:
        max_dist = metrics.get('max_distance', 0)
        st.metric("Max Distance", f"{max_dist:.2f}", help="Objective: minimize this value")
    with cols[1]:
        avg_dist = metrics.get('average_distance', 0)
        st.metric("Avg Distance", f"{avg_dist:.2f}")
    with cols[2]:
        min_dist = metrics.get('min_distance', 0)
        st.metric("Min Distance", f"{min_dist:.2f}")
    with cols[3]:
        n_fac = metrics.get('num_facilities', 0)
        st.metric("Facilities", int(n_fac))
    
    with st.expander("Detailed Metrics"):
        st.write(f"**Demand Points Served:** {metrics.get('num_demand_points', 0)}")
        st.write(f"**Std Deviation:** {metrics.get('std_distance', 0):.2f}")


def _display_pmedian_metrics(metrics: dict):
    """Display P-Median-specific metrics."""
    obj_type = metrics.get('objective_type', 'total')
    
    st.markdown("##### Distance Metrics")
    cols = st.columns(4)
    
    with cols[0]:
        if obj_type == 'average':
            avg_dist = metrics.get('average_distance', 0)
            st.metric("Avg Weighted Distance", f"{avg_dist:.2f}", help="Objective value")
        else:
            total_dist = metrics.get('total_weighted_distance', 0)
            st.metric("Total Weighted Distance", f"{total_dist:.2f}", help="Objective value")
    with cols[1]:
        if obj_type == 'average':
            total_dist = metrics.get('total_weighted_distance', 0)
            st.metric("Total Distance", f"{total_dist:.2f}")
        else:
            avg_dist = metrics.get('average_distance', 0)
            st.metric("Avg Distance", f"{avg_dist:.2f}")
    with cols[2]:
        max_dist = metrics.get('max_distance', 0)
        st.metric("Max Distance", f"{max_dist:.2f}")
    with cols[3]:
        n_fac = metrics.get('num_facilities', 0)
        st.metric("Facilities", int(n_fac))
    
    with st.expander("Detailed Metrics"):
        st.write(f"**Objective Type:** {obj_type.title()}")
        st.write(f"**Demand Points Served:** {metrics.get('num_demand_points', 0)}")
        st.write(f"**Total Demand Weight:** {metrics.get('total_demand_weight', 0):.2f}")
        
        # Variant-specific
        if 'budget_used' in metrics:
            st.write(f"**Budget Used:** {metrics.get('budget_used', 0):.2f}")
        if 'capacity_utilization' in metrics:
            cap_util = metrics.get('capacity_utilization', {})
            if isinstance(cap_util, dict) and cap_util:
                st.write("**Capacity Utilization by Facility:**")
                for fac_id, util in cap_util.items():
                    if util is not None:
                        st.write(f"  - Facility {fac_id}: {util*100:.1f}%")
        
        # Validation warnings
        if 'violation_count' in metrics and metrics['violation_count'] > 0:
            st.warning(f"⚠️ {metrics['violation_count']} assignment violation(s) detected")


def _display_generic_metrics(metrics: dict):
    """Display generic metrics for unknown problem types."""
    with st.expander("Detailed Metrics", expanded=True):
        for key, value in metrics.items():
            if key in ('objective_value', 'objective_name'):
                continue  # Already displayed in header
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                st.write(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
            elif isinstance(value, dict):
                st.write(f"**{key.replace('_', ' ').title()}:** (complex)")
            else:
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")


# ============================================================================
# PERFORMANCE: Use fragment for map to prevent full page reruns
# ============================================================================

@st.fragment
def render_map_fragment():
    """Render map in a fragment to avoid full page reruns"""
    if not st.session_state.problem_state["data"] and not st.session_state.problem_state["solution"]:
        st.info("Upload data or start a conversation to see visualizations")
        return
    
    try:
        # Get visualization config if problem type is known
        viz_config = None
        if st.session_state.problem_state["problem_type"]:
            problem_solver = problem_registry.get_problem(st.session_state.problem_state["problem_type"])
            if problem_solver:
                viz_config = problem_solver.get_visualization_config()
        
        # Optional UI toggle to show service areas when radius is available
        try:
            current_problem = (st.session_state.problem_state["problem_type"] or "").lower()
            params = st.session_state.problem_state.get("parameters", {})
            sol = st.session_state.problem_state.get("solution", {}) or {}
            metrics = sol.get("metrics", {})
            service_radius = params.get("service_radius") or metrics.get("service_radius")
            if service_radius is not None and current_problem in ["mclp", "lscp"]:
                show_radius = st.checkbox("Show service radius", value=True, key="show_service_radius_frag")
                if viz_config is None:
                    viz_config = {}
                viz_config["show_service_areas"] = bool(show_radius)
        except Exception:
            pass

        # Generate candidate sites once for visualization if missing and persist
        try:
            data_items = st.session_state.problem_state["data"]
            data_processor = st.session_state.data_processor
            has_demand_viz = False
            has_candidates_viz = False
            for fname, fgdf in data_items.items():
                dtype = data_processor.identify_data_type(fgdf)
                if dtype == "demand_points" or "demand" in fname.lower():
                    has_demand_viz = True
                if dtype == "candidate_sites" or any(w in fname.lower() for w in ["candidate", "site", "facility"]):
                    has_candidates_viz = True

            if has_demand_viz and not has_candidates_viz and "generated_candidates" not in data_items:
                demand_gdf_viz = None
                for fname, fgdf in data_items.items():
                    dtype = data_processor.identify_data_type(fgdf)
                    if dtype == "demand_points" or "demand" in fname.lower():
                        demand_gdf_viz = fgdf
                        break
                if demand_gdf_viz is not None and len(demand_gdf_viz) > 0:
                    num_sites = st.session_state.get("generated_sites_count", 100)
                    random_seed = st.session_state.get("generated_sites_seed", None)
                    generated_candidates_viz = data_processor.generate_candidate_sites(
                        demand_gdf_viz,
                        num_sites=num_sites,
                        random_seed=random_seed
                    )
                    st.session_state.problem_state["data"]["generated_candidates"] = generated_candidates_viz
        except Exception as viz_gen_err:
            logger.warning(f"Could not auto-generate candidate sites for visualization: {viz_gen_err}")
        
        # Map data to expected format for visualizer
        data_processor = st.session_state.data_processor
        mapped_data = {}
        
        for file_name, gdf in st.session_state.problem_state["data"].items():
            data_type = data_processor.identify_data_type(gdf)
            
            if data_type == "demand_points" or "demand" in file_name.lower():
                mapped_data["demand_points"] = gdf
            elif data_type == "candidate_sites" or any(word in file_name.lower() for word in ['candidate', 'site', 'facility']):
                mapped_data["candidate_sites"] = gdf
            elif "demand_points" not in mapped_data:
                mapped_data["demand_points"] = gdf
            elif "candidate_sites" not in mapped_data:
                mapped_data["candidate_sites"] = gdf
        
        # Prepare parameters with service radius unit from solution
        parameters = st.session_state.problem_state.get("parameters", {}).copy()
        solution = st.session_state.problem_state["solution"]
        if solution and "service_radius_unit" in solution:
            parameters["service_radius_unit"] = solution["service_radius_unit"]
        
        # Choose map renderer based on user preference
        if st.session_state.get("map_renderer", "pydeck") == "pydeck":
            # Get basemap style from session state
            basemap_style = st.session_state.get("basemap_style", "light")
            
            deck = st.session_state.pydeck_visualizer.create_map(
                data=mapped_data,
                solution=solution,
                problem_type=st.session_state.problem_state["problem_type"],
                viz_config=viz_config,
                parameters=parameters,
                constraints=st.session_state.problem_state.get("constraints", {}),
                basemap_style=basemap_style
            )
            st.pydeck_chart(deck, height=500, width="stretch")
        else:
            map_obj = st.session_state.map_visualizer.create_map(
                data=mapped_data,
                solution=solution,
                problem_type=st.session_state.problem_state["problem_type"],
                viz_config=viz_config,
                parameters=parameters,
                constraints=st.session_state.problem_state.get("constraints", {}),
                raster_data=st.session_state.get("raster_data", {})
            )
            st_folium(map_obj, width=700, height=500, key="map_frag")
    
    except Exception as e:
        st.error(f"Error creating map: {e}")
        logger.error(f"Map error: {e}", exc_info=True)
    


with col1:
    st.subheader("Conversation")
    
    # Chat container
    chat_container = st.container(height=500)
    
    with chat_container:
        # Display message history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Describe your spatial optimization problem..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get data summary for context
        data_summary = None
        if st.session_state.problem_state["data"]:
            data_summary = {}
            for name, gdf in st.session_state.problem_state["data"].items():
                try:
                    dtypes = {c: str(gdf[c].dtype) for c in gdf.columns if c != 'geometry'}
                except Exception:
                    dtypes = {}
                
                # Detect special columns
                data_processor = st.session_state.data_processor
                capacity_cols = data_processor.identify_capacity_columns(gdf)
                cost_cols = data_processor.identify_cost_columns(gdf)
                demand_cols = data_processor.identify_demand_columns(gdf)
                
                data_summary[name] = {
                    "num_features": len(gdf),
                    "geometry_type": gdf.geometry.type.unique()[0] if len(gdf) > 0 else "Unknown",
                    "columns": [c for c in gdf.columns if c != 'geometry'],
                    "dtypes": dtypes,
                    "bounds": gdf.total_bounds.tolist() if len(gdf) > 0 else [],
                    "capacity_columns": capacity_cols,
                    "cost_columns": cost_cols,
                    "demand_columns": demand_cols
                }
        
        # Call conversation manager
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.conversation_manager.chat(
                    user_message=prompt,
                    conversation_history=st.session_state.messages[:-1],  # Exclude current message
                    problem_state=st.session_state.problem_state,
                    uploaded_data_summary=data_summary
                )
            except Exception as e:
                st.error(f"Error communicating with AI: {e}")
                result = {
                    "response": f"I encountered an error: {str(e)}",
                    "actions": [],
                    "updated_state": st.session_state.problem_state
                }
        
        # Update state
        st.session_state.problem_state = result["updated_state"]
        
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"]
        })
        
        # Handle actions (e.g., optimization trigger)
        if result["actions"]:
            logger.info(f"App: Processing {len(result['actions'])} action(s): {[a.get('action', 'unknown') for a in result['actions']]}")
            
            # Track processed actions to prevent duplicates
            processed_actions = set()
            
            for i, action in enumerate(result["actions"]):
                if action["action"] == "optimize":
                    # Create a unique key for this action to prevent duplicates
                    action_key = f"{action.get('problem_type', 'unknown')}_{action.get('parameters', {}).get('n_facilities', 'unknown')}_{action.get('parameters', {}).get('service_radius', 'unknown')}"
                    
                    if action_key in processed_actions:
                        logger.warning(f"App: Skipping duplicate optimization action: {action_key}")
                        continue
                    
                    processed_actions.add(action_key)
                    logger.info(f"App: Processing optimization action {i+1}/{len(result['actions'])}: {action_key}")
                    with st.spinner("Running optimization..."):
                        try:
                            # Get problem solver
                            problem_solver = problem_registry.get_problem(action["problem_type"])
                            
                            if not problem_solver:
                                st.error(f"Problem type '{action['problem_type']}' not found")
                                continue
                            
                            # Prepare data mapping
                            data_dict = {}
                            data_processor = st.session_state.data_processor
                            
                            # Try to intelligently map uploaded files to required data
                            for file_name, gdf in st.session_state.problem_state["data"].items():
                                data_type = data_processor.identify_data_type(gdf)
                                
                                if data_type == "demand_points" or "demand" in file_name.lower():
                                    data_dict["demand_points"] = gdf
                                elif data_type == "candidate_sites" or any(word in file_name.lower() for word in ['candidate', 'site', 'facility']):
                                    data_dict["candidate_sites"] = gdf
                            
                            # Fallback: if we still don't have both types, make educated guesses
                            if "demand_points" not in data_dict and "candidate_sites" not in data_dict:
                                # If we have exactly 2 datasets, assume first is demand, second is candidates
                                data_files = list(st.session_state.problem_state["data"].items())
                                if len(data_files) == 2:
                                    data_dict["demand_points"] = data_files[0][1]
                                    data_dict["candidate_sites"] = data_files[1][1]
                                elif len(data_files) == 1:
                                    # Single dataset - assume it's demand points
                                    data_dict["demand_points"] = data_files[0][1]
                            elif "demand_points" not in data_dict:
                                # We have candidates but no demand - use first remaining dataset as demand
                                remaining_files = [(name, gdf) for name, gdf in st.session_state.problem_state["data"].items() 
                                                 if name not in [k for k, v in data_dict.items() if v is not None]]
                                if remaining_files:
                                    data_dict["demand_points"] = remaining_files[0][1]
                            elif "candidate_sites" not in data_dict:
                                # We have demand but no candidates - use first remaining dataset as candidates
                                remaining_files = [(name, gdf) for name, gdf in st.session_state.problem_state["data"].items() 
                                                 if name not in [k for k, v in data_dict.items() if v is not None]]
                                if remaining_files:
                                    data_dict["candidate_sites"] = remaining_files[0][1]
                            
                            # Generate candidate sites if we have demand but no candidates
                            if "demand_points" in data_dict and "candidate_sites" not in data_dict:
                                logger.info("No candidate sites found - generating random sites within demand extent")
                                try:
                                    # Get generation parameters from session state
                                    num_sites = st.session_state.get("generated_sites_count", 100)
                                    random_seed = st.session_state.get("generated_sites_seed", None)
                                    
                                    # Generate candidate sites
                                    generated_candidates = data_processor.generate_candidate_sites(
                                        data_dict["demand_points"], 
                                        num_sites=num_sites, 
                                        random_seed=random_seed
                                    )
                                    data_dict["candidate_sites"] = generated_candidates
                                    # Persist generated candidates so they appear on the map and in state
                                    try:
                                        st.session_state.problem_state["data"]["generated_candidates"] = generated_candidates
                                    except Exception as persist_error:
                                        logger.warning(f"Could not persist generated candidate sites to session state: {persist_error}")
                                    
                                    logger.info(f"Generated {num_sites} candidate sites with seed {random_seed}")
                                    
                                    # Add info message to chat
                                    seed_info = f" (seed: {random_seed})" if random_seed is not None else ""
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"Generated {num_sites} random candidate sites within demand extent{seed_info}."
                                    })
                                    
                                except Exception as gen_error:
                                    logger.error(f"Failed to generate candidate sites: {gen_error}")
                                    st.error(f"Failed to generate candidate sites: {gen_error}")
                                    continue
                            
                            # Auto-detect and add variant-specific parameters from data
                            parameters = action.get("parameters", {}).copy()
                            
                            # Only auto-detect data if variant is explicitly requested by user
                            # Check if capacitated variant and no capacities provided
                            if parameters.get("variant") == "capacitated" and "capacities" not in parameters:
                                # First try to get capacity data from candidate sites
                                if "candidate_sites" in data_dict:
                                    capacity_data = data_processor.extract_capacity_data(data_dict["candidate_sites"])
                                    if capacity_data:
                                        parameters["capacities"] = capacity_data
                                        logger.info(f"Auto-detected capacity data from candidate sites: {len(capacity_data)} values")
                                    else:
                                        # If no capacity data in candidate sites, calculate based on demand dataset population
                                        if "demand_points" in data_dict:
                                            demand_population = data_processor.extract_demand_data(data_dict["demand_points"])
                                            if demand_population:
                                                total_demand = sum(demand_population)
                                                n_facilities = parameters.get("n_facilities", len(data_dict["candidate_sites"]))
                                                # Distribute total demand among facilities
                                                avg_capacity = total_demand / n_facilities
                                                # Create capacity array for all candidate sites
                                                capacity_data = [avg_capacity] * len(data_dict["candidate_sites"])
                                                parameters["capacities"] = capacity_data
                                                logger.info(f"Calculated capacity data based on demand population: {len(capacity_data)} values, avg capacity: {avg_capacity:.2f}")
                                            else:
                                                logger.warning("Capacitated variant requested but no population data found in demand dataset")
                                        else:
                                            logger.warning("Capacitated variant requested but no demand points data available")
                            
                            # Check if budget variant and no costs provided
                            # Only auto-detect costs if variant is explicitly set to budget
                            if parameters.get("variant") == "budget" and "facility_costs" not in parameters:
                                if "candidate_sites" in data_dict:
                                    cost_data = data_processor.extract_cost_data(data_dict["candidate_sites"])
                                    if cost_data:
                                        parameters["facility_costs"] = cost_data
                                        logger.info(f"Auto-detected cost data: {len(cost_data)} values")
                                    else:
                                        logger.warning("Budget variant requested but no cost data found in candidate sites")
                                        # Remove budget variant if no cost data available
                                        parameters["variant"] = "base" if parameters.get("problem_type") == "p-median" else "classical"
                                        logger.info(f"Reverted to {'base' if parameters.get('problem_type') == 'p-median' else 'classical'} variant due to missing cost data")
                            
                            # Add default weights if needed (use a non-conflicting column name)
                            if "demand_points" in data_dict:
                                data_dict["demand_points"] = data_processor.add_default_weights(data_dict["demand_points"], weight_column='default_weight')
                            
                            # Validate required data
                            required_data = problem_solver.get_required_data()
                            missing_data = [k for k, v in required_data.items() if v.get('required') and k not in data_dict]
                            
                            if missing_data:
                                error_msg = f"Missing required data: {', '.join(missing_data)}"
                                st.error(error_msg)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"{error_msg}. Please upload the required data files."
                                })
                                st.rerun()
                                continue
                            
                            # Solve with performance monitoring
                            logger.info(f"App: Solving with parameters: {parameters}")
                            start_time = time.time()
                            
                            try:
                                solution = problem_solver.solve(
                                    data=data_dict,
                                    parameters=parameters,
                                    constraints=action.get("constraints", {}),
                                    distance_metric=action.get("distance_metric", "euclidean")
                                )
                                
                                solve_time = time.time() - start_time
                                logger.info(f"Optimization completed in {solve_time:.2f} seconds")
                                
                                # Log performance metrics
                                if solution.get('status') in ['optimal', 'feasible']:
                                    logger.info(f"Solution status: {solution.get('status')}")
                                    logger.info(f"Objective value: {solution.get('objective_value', 'N/A')}")
                                    logger.info(f"Selected facilities: {len(solution.get('selected_facilities', []))}")
                                    
                            except Exception as solve_error:
                                solve_time = time.time() - start_time
                                logger.error(f"Optimization failed after {solve_time:.2f} seconds: {solve_error}")
                                raise
                            
                            st.session_state.problem_state["solution"] = solution
                            st.session_state.problem_state["solution_history"].append(solution)
                            
                            # Generate explanation
                            if solution.get('status') != 'error':
                                # Build kwargs based on solver signature to avoid unexpected kwargs
                                explain_sig = inspect.signature(problem_solver.explain_solution)
                                explain_kwargs = {
                                    "solution": solution,
                                    "data": data_dict,
                                    "detail_level": "standard",
                                }
                                if "objective_type" in explain_sig.parameters:
                                    explain_kwargs["objective_type"] = parameters.get("objective", "total")
                                explanation = problem_solver.explain_solution(**explain_kwargs)
                                
                                # Add explanation to chat
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"**Optimization Complete.**\n\n{explanation}"
                                })
                                
                                st.success("Optimization completed! Check the map for results.")
                            else:
                                error_msg = f"Optimization failed: {solution.get('error', 'Unknown error')}"
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": error_msg
                                })
                                st.error(error_msg)
                        
                        except Exception as e:
                            error_msg = f"Optimization error: {str(e)}"
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                            st.error(error_msg)
                            logger.error(f"Optimization error: {e}", exc_info=True)
        
        st.rerun()

with col2:
    st.subheader("Visualization")
    
    # Use fragment for map rendering to prevent full page reruns
    render_map_fragment()
    
    # Metrics dashboard (outside fragment to ensure it updates)
    if st.session_state.problem_state["solution"]:
        st.divider()
        st.subheader("Solution Metrics")
        
        solution = st.session_state.problem_state["solution"]
        problem_type = (st.session_state.problem_state.get("problem_type") or "").lower()
        
        if solution.get('status') in ['optimal', 'feasible']:
            metrics = solution.get("metrics", {})
            
            # Display problem-specific key metrics
            _display_problem_metrics(problem_type, solution, metrics)
        else:
            st.warning(f"Solution status: {solution.get('status', 'Unknown')}")
            if solution.get('error'):
                st.error(solution['error'])
    
    # Export options (outside fragment to avoid key conflicts)
    if st.session_state.problem_state["solution"]:
        solution = st.session_state.problem_state["solution"]
        if solution.get('status') in ['optimal', 'feasible']:
            st.divider()
            st.subheader("Export Solution")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("Export GeoJSON"):
                    try:
                        geojson_data = st.session_state.export_handler.export_solution_geojson(
                            solution, st.session_state.problem_state["data"]
                        )
                        st.download_button(
                            "Download GeoJSON",
                            geojson_data,
                            "solution.geojson",
                            "application/geo+json"
                        )
                    except Exception as e:
                        st.error(f"Export error: {e}")
            
            with export_col2:
                if st.button("Export CSV"):
                    try:
                        csv_data = st.session_state.export_handler.export_solution_csv(solution)
                        st.download_button(
                            "Download CSV",
                            csv_data,
                            "solution.csv",
                            "text/csv"
                        )
                    except Exception as e:
                        st.error(f"Export error: {e}")
            
            with export_col3:
                if st.button("Generate PDF Report"):
                    try:
                        problem_metadata = {}
                        if st.session_state.problem_state["problem_type"]:
                            problem_solver = problem_registry.get_problem(
                                st.session_state.problem_state["problem_type"]
                            )
                            if problem_solver:
                                problem_metadata = problem_solver.get_metadata()
                        
                        pdf_data = st.session_state.export_handler.generate_pdf_report(
                            solution,
                            problem_metadata,
                            st.session_state.problem_state["parameters"]
                        )
                        st.download_button(
                            "Download PDF",
                            pdf_data,
                            "solution_report.pdf",
                            "application/pdf"
                        )
                    except Exception as e:
                        st.error(f"PDF generation error: {e}")
    
    # Quick start guide when no data
    if not st.session_state.problem_state["data"] and not st.session_state.problem_state["solution"]:
        with st.expander("Quick Start Guide"):
            st.markdown("""
            **Step 1: Upload Data**
            - Upload demand points (e.g., population centers) - **Required**
            - Upload candidate sites (e.g., potential facility locations) - **Optional**
            
            **New: Automatic Candidate Generation**
            If you only upload demand data, the system will automatically generate 100 random candidate sites within your demand extent. Adjust the count and seed in the sidebar.
            
            **Step 2: Describe Your Problem**
            Examples:
            - "I need to locate 5 fire stations to minimize response times"
            - "Where should I place 3 warehouses to minimize average distance?"
            - "I want to maximize coverage within 5km using 4 facilities"
            
            **Step 3: Review and Refine**
            - Check the solution on the map
            - Ask questions about the results
            - Try different scenarios
            """)

# Footer
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("Spatial Optimization Agent")
with col_f2:
    st.caption(f"Powered by Google Gemini")
with col_f3:
    st.caption("")

