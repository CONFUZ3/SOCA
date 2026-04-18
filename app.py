import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
import pydeck as pdk
import geopandas as gpd
from pathlib import Path
import os
import re
import logging
import sys
import time
import hashlib
import json

# Set up logging — force UTF-8 so activity-log glyphs (✓ … • ✗) don't crash
# the Windows cp1252 console. errors="replace" is a belt-and-braces fallback.
try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_log_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_stream_handler = logging.StreamHandler(stream=sys.stderr)
_stream_handler.setFormatter(_log_fmt)
_file_handler = logging.FileHandler('spopt_app.log', encoding='utf-8')
_file_handler.setFormatter(_log_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
logger = logging.getLogger(__name__)

# Imports from our modules
from agent.soca_agent import SOCAAgent
from solvers.registry import problem_registry
from utils.data_processor import DataProcessor
from utils.visualizer import MapVisualizer
from utils.pydeck_visualizer import PyDeckVisualizer
from utils.export_handler import ExportHandler
from utils.aoi_selector import render_aoi_selector, aoi_to_boundary_gdf
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
    
    if "problem_state" not in st.session_state:
        st.session_state.problem_state = {
            "problem_type": None,
            "parameters": {},
            "constraints": {},
            "data": {},
            "solution": None,
            "solution_history": [],
            "aoi": None,
            "aoi_confirmed": False,
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
        
        st.session_state.conversation_manager = SOCAAgent(
            api_key=api_key,
            problem_registry=problem_registry
        )
    
    if "data_processor" not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    
    if "map_visualizer" not in st.session_state:
        st.session_state.map_visualizer = MapVisualizer()
    
    if "export_handler" not in st.session_state:
        st.session_state.export_handler = ExportHandler()
    
    if "data_fetcher" not in st.session_state:
        st.session_state.data_fetcher = None  # Lazy-initialised on first use

    if "network_manager" not in st.session_state:
        from utils.network_manager import NetworkManager
        st.session_state.network_manager = NetworkManager()

initialize_session_state()

# Inject NetworkManager into problem_state on every rerun so soca_agent can access it
st.session_state.problem_state["_network_manager"] = st.session_state.network_manager


def _reset_to_aoi_step() -> None:
    """Fresh start: clear data, solution, chat and return to AOI selection."""
    st.session_state.messages = []
    st.session_state.problem_state = {
        "problem_type": None,
        "parameters": {},
        "constraints": {},
        "data": {},
        "solution": None,
        "solution_history": [],
        "aoi": None,
        "aoi_confirmed": False,
        "_network_manager": st.session_state.network_manager,
    }
    st.session_state.raster_data = {}
    # Clear AOI selector widget state
    for k in (
        "_aoi_search_result", "_aoi_current_geom", "_aoi_current_name",
        "_aoi_current_source", "_aoi_search_input",
    ):
        if k in st.session_state:
            st.session_state[k] = None if "input" not in k else ""
    st.session_state["_aoi_map_key"] = st.session_state.get("_aoi_map_key", 0) + 1


# ============================================================================
# AOI GATE — Step 1 of the flow. Until AOI is confirmed, nothing else renders.
# ============================================================================
if not st.session_state.problem_state.get("aoi_confirmed"):
    st.title("Spatial Optimization Conversational Agent")
    confirmed = render_aoi_selector()
    if confirmed is not None:
        aoi_gdf = aoi_to_boundary_gdf(confirmed)
        st.session_state.problem_state["aoi"] = confirmed
        st.session_state.problem_state["aoi_confirmed"] = True
        st.session_state.problem_state["data"]["boundary_aoi"] = aoi_gdf
        # Seed a scoped welcome message
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                f"Great — your area of interest is **{confirmed['name']}** "
                f"({confirmed['area_km2']:,.1f} km²).\n\n"
                "Now tell me what you want to optimize. Examples:\n"
                "- *Place 5 hospitals here to minimize travel distance*\n"
                "- *Maximize clinic coverage within a 2 km radius using 4 facilities*\n"
                "- *How many fire stations do I need to cover every block within 3 km?*\n\n"
                "I can also fetch population grids and existing facility locations "
                "for this area — just ask."
            ),
        }]
        st.rerun()
    st.stop()


# ============================================================================
# AOI HEADER — persistent chip showing current AOI with Change AOI button
# ============================================================================
_aoi = st.session_state.problem_state.get("aoi") or {}
_hdr_cols = st.columns([6, 1])
with _hdr_cols[0]:
    st.markdown(
        f"**AOI:** {_aoi.get('name', '—')} · "
        f"{_aoi.get('area_km2', 0):,.1f} km² · "
        f"source `{_aoi.get('source', '—')}`"
    )
with _hdr_cols[1]:
    if st.button("Change AOI", use_container_width=True, help="Clears data, solutions, and chat"):
        if st.session_state.problem_state.get("solution") or st.session_state.problem_state.get("data"):
            st.session_state["_aoi_change_pending"] = True
        else:
            _reset_to_aoi_step()
            st.rerun()

if st.session_state.get("_aoi_change_pending"):
    with st.container(border=True):
        st.warning(
            "Changing the AOI will clear loaded data, the current solution, "
            "and the chat history. This cannot be undone."
        )
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("Confirm change", type="primary"):
                st.session_state["_aoi_change_pending"] = False
                _reset_to_aoi_step()
                st.rerun()
        with c2:
            if st.button("Cancel"):
                st.session_state["_aoi_change_pending"] = False
                st.rerun()

st.divider()

# Sidebar
with st.sidebar:
    st.title("Spatial Optimization")
    
    st.divider()
    
    # File upload section
    st.subheader("Upload Data")
    st.markdown("Upload geospatial data files (GeoJSON, Shapefile, CSV)")
    st.caption("🌐 Or just describe your problem with a **location name** and I'll fetch data automatically!")
    
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
                
                # Add sample values for LLM context (first non-null value per column)
                sample_values = {}
                column_stats = {}
                for col in gdf.columns:
                    if col.lower() in ['geometry', 'shape']:
                        continue
                    try:
                        non_null = gdf[col].dropna()
                        if len(non_null) > 0:
                            sample_values[col] = non_null.iloc[0]
                            # Add stats for numeric columns
                            if gdf[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                                column_stats[col] = {
                                    'mean': float(gdf[col].mean()),
                                    'max': float(gdf[col].max())
                                }
                    except Exception:
                        pass
                
                data_summary[name] = {
                    "num_features": len(gdf),
                    "geometry_type": gdf.geometry.type.unique()[0] if len(gdf) > 0 else "Unknown",
                    "columns": [c for c in gdf.columns if c != 'geometry'],
                    "dtypes": dtypes,
                    "bounds": gdf.total_bounds.tolist() if len(gdf) > 0 else [],
                    "capacity_columns": capacity_cols,
                    "cost_columns": cost_cols,
                    "demand_columns": demand_cols,
                    "sample_values": sample_values,
                    "column_stats": column_stats
                }
            try:
                with st.spinner("Syncing uploaded data with AI..."):
                    # Propagate generated-sites settings into problem_state for tools
                    st.session_state.problem_state["_generated_sites_count"] = st.session_state.get("generated_sites_count", 100)
                    st.session_state.problem_state["_generated_sites_seed"] = st.session_state.get("generated_sites_seed", None)
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
            if data_type == "candidate_sites" or any(word in name.lower() for word in ['candidate', 'site', 'facility']):
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
            "solution_history": [],
            "aoi": st.session_state.problem_state.get("aoi"),
            "aoi_confirmed": st.session_state.problem_state.get("aoi_confirmed", False),
        }
        st.rerun()

# Main content area

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
            label = _get_metric_label(obj_name)
            st.metric(label, f"{obj_val:.2f}")
    
    with metric_cols[1]:
        st.metric("Solution Status", solution.get('status', 'Unknown').title())
    
    with metric_cols[2]:
        st.metric("Solution Time", f"{solution.get('solution_time', 0):.2f}s")
    
    # Get distance unit from solution
    distance_unit = solution.get('service_radius_unit', 'm')
    
    # Problem-specific key metrics
    if problem_type == "lscp":
        _display_lscp_metrics(metrics, distance_unit)
    elif problem_type == "mclp":
        _display_mclp_metrics(metrics, solution, distance_unit)
    elif problem_type == "p-center":
        _display_pcenter_metrics(metrics, distance_unit)
    elif problem_type == "p-median":
        _display_pmedian_metrics(metrics, distance_unit)
    else:
        # Generic fallback
        _display_generic_metrics(metrics)


def _get_metric_label(key: str) -> str:
    """Get human-friendly label for a metric key."""
    labels = {
        'coverage_percentage': 'Coverage Percentage',
        'num_facilities': 'Facilities Selected',
        'average_distance_covered': 'Avg Distance (covered)',
        'max_distance': 'Max Distance',
        'average_distance': 'Average Distance',
        'total_weighted_distance': 'Total Weighted Distance',
        'average_weighted_distance': 'Avg Weighted Distance',
        'covered_demand': 'Covered Demand',
        'served_demand': 'Served Demand',
        'expected_covered_demand': 'Expected Coverage',
    }
    return labels.get(key, key.replace('_', ' ').title())


def _format_distance(value: float, unit: str) -> str:
    """Format a distance value with its unit consistently."""
    # Mapping to pretty unit labels
    pretty_units = {
        'm': 'm',
        'km': 'km',
        'mi': 'miles',
        'ft': 'ft',
        'yd': 'yards',
        'nm': 'nm'
    }
    unit_label = pretty_units.get(unit.lower().strip(), unit)
    return f"{value:.2f} {unit_label}"


def _display_lscp_metrics(metrics: dict, unit: str = 'm'):
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
        st.metric("Service Radius", _format_distance(radius, unit))
    
    # Additional details
    with st.expander("Detailed Metrics"):
        st.write(f"**Total Demand Points:** {metrics.get('total_demand_points', 0)}")
        st.write(f"**Covered Points:** {metrics.get('num_covered_points', 0)}")
        st.write(f"**Uncovered Points:** {metrics.get('num_uncovered_points', 0)}")
        st.write(f"**Average Distance:** {_format_distance(metrics.get('average_distance', 0), unit)}")
        st.write(f"**Max Distance:** {_format_distance(metrics.get('max_distance', 0), unit)}")


def _display_mclp_metrics(metrics: dict, solution: dict, unit: str = 'm'):
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
        st.metric("Service Radius", _format_distance(radius, unit))
    
    # Variant-specific metrics
    with st.expander("Detailed Metrics"):
        st.write(f"**Variant:** {variant.title()}")
        st.write(f"**Facilities Selected:** {metrics.get('num_facilities', 0)}")
        st.write(f"**Total Demand:** {metrics.get('total_demand', 0):.1f}")
        st.write(f"**Covered Points:** {metrics.get('num_covered_points', 0)}")
        st.write(f"**Avg Distance (covered):** {_format_distance(metrics.get('average_distance_covered', 0), unit)}")
        
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


def _display_pcenter_metrics(metrics: dict, unit: str = 'm'):
    """Display P-Center-specific metrics."""
    st.markdown("##### Distance Metrics (Minimax)")
    cols = st.columns(4)
    
    with cols[0]:
        max_dist = metrics.get('max_distance', 0)
        st.metric("Max Distance", _format_distance(max_dist, unit), help="Objective: minimize this value")
    with cols[1]:
        avg_dist = metrics.get('average_distance', 0)
        st.metric("Avg Distance", _format_distance(avg_dist, unit))
    with cols[2]:
        min_dist = metrics.get('min_distance', 0)
        st.metric("Min Distance", _format_distance(min_dist, unit))
    with cols[3]:
        n_fac = metrics.get('num_facilities', 0)
        st.metric("Facilities", int(n_fac))
    
    with st.expander("Detailed Metrics"):
        st.write(f"**Demand Points Served:** {metrics.get('num_demand_points', 0)}")
        st.write(f"**Std Deviation:** {_format_distance(metrics.get('std_distance', 0), unit)}")


def _display_pmedian_metrics(metrics: dict, unit: str = 'm'):
    """Display P-Median-specific metrics."""
    obj_type = metrics.get('objective_type', 'total')
    
    st.markdown("##### Distance Metrics")
    cols = st.columns(4)
    
    with cols[0]:
        if obj_type == 'average':
            avg_dist = metrics.get('average_distance', 0)
            st.metric("Avg Weighted Distance", _format_distance(avg_dist, unit), help="Objective value")
        else:
            total_dist = metrics.get('total_weighted_distance', 0)
            st.metric("Total Weighted Distance", _format_distance(total_dist, unit), help="Objective value")
    with cols[1]:
        if obj_type == 'average':
            total_dist = metrics.get('total_weighted_distance', 0)
            st.metric("Total Distance", _format_distance(total_dist, unit))
        else:
            avg_dist = metrics.get('average_distance', 0)
            st.metric("Avg Distance", _format_distance(avg_dist, unit))
    with cols[2]:
        max_dist = metrics.get('max_distance', 0)
        st.metric("Max Distance", _format_distance(max_dist, unit))
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
        # (Moved to specific renderer blocks to allow for better UI placement)
        
        # Generate candidate sites once for visualization if missing and persist
        try:
            data_items = st.session_state.problem_state["data"]
            data_processor = st.session_state.data_processor

            # Categorize datasets (reuse same logic as optimize handler)
            _viz_boundary_keys = set()
            _viz_poi_keys = set()
            _viz_demand_keys = set()
            for _fname, _fgdf in data_items.items():
                _src = _fgdf.attrs.get("source", "")
                _fkey = _fname.lower()
                if _fkey.startswith("boundary_") or _src in (
                    "auto_fetched", "osmnx",
                    "photon_bbox_fallback",
                    "nominatim", "nominatim_bbox_fallback",
                    "gadm",
                ):
                    if len(_fgdf) > 0 and _fgdf.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon"):
                        _viz_boundary_keys.add(_fname)
                        continue
                if "_facilities_" in _fkey or any(
                    _fkey.startswith(_c + "_") for _c in [
                        "health", "education", "food", "finance",
                        "fire_station", "police", "library", "generated",
                    ]
                ) or _fkey == "generated_candidates":
                    _viz_poi_keys.add(_fname)
                    continue
                _dtype = data_processor.identify_data_type(_fgdf)
                if _dtype == "demand_points" or "demand" in _fkey:
                    _viz_demand_keys.add(_fname)
                elif _dtype == "candidate_sites" or any(
                    w in _fkey for w in ["candidate", "site", "facility"]
                ):
                    _viz_poi_keys.add(_fname)
                else:
                    _viz_demand_keys.add(_fname)

            has_demand_viz    = bool(_viz_demand_keys)
            has_candidates_viz = bool(_viz_poi_keys)

            if has_demand_viz and not has_candidates_viz and "generated_candidates" not in data_items:
                demand_gdf_viz = None
                for _fname in _viz_demand_keys:
                    demand_gdf_viz = data_items[_fname]
                    break

                boundary_gdf_viz = None
                for _fname in _viz_boundary_keys:
                    boundary_gdf_viz = data_items[_fname]
                    break

                sampling_gdf_viz = boundary_gdf_viz if boundary_gdf_viz is not None else demand_gdf_viz

                if sampling_gdf_viz is not None and len(sampling_gdf_viz) > 0:
                    num_sites = st.session_state.get("generated_sites_count", 100)
                    random_seed = st.session_state.get("generated_sites_seed", None)
                    generated_candidates_viz = data_processor.generate_candidate_sites(
                        sampling_gdf_viz,
                        num_sites=num_sites,
                        random_seed=random_seed
                    )
                    st.session_state.problem_state["data"]["generated_candidates"] = generated_candidates_viz
                    _viz_poi_keys.add("generated_candidates")
        except Exception as viz_gen_err:
            logger.warning(f"Could not auto-generate candidate sites for visualization: {viz_gen_err}")

        # Map data to expected format for visualizer (skip boundary/polygon datasets)
        data_processor = st.session_state.data_processor
        mapped_data = {}

        for file_name, gdf in st.session_state.problem_state["data"].items():
            # Skip boundary polygons — they are not demand or candidates
            if file_name in _viz_boundary_keys:
                continue

            if file_name in _viz_poi_keys:
                mapped_data["candidate_sites"] = gdf
            elif file_name in _viz_demand_keys:
                mapped_data["demand_points"] = gdf
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
            if viz_config is None:
                viz_config = {}
                
            # Layer Selector Popover
            # Placed in a column to appear as a small button above the map
            ls_col1, ls_col2 = st.columns([0.2, 0.8])
            with ls_col1:
                with st.popover("🗺️ Layers", help="Toggle map layers"):
                    st.caption("Layer Visibility")
                    viz_config["show_demand"] = st.checkbox("Demand Points", value=True, key="pd_show_demand")
                    viz_config["show_candidates"] = st.checkbox("Candidate Sites", value=True, key="pd_show_candidates")
                    viz_config["show_facilities"] = st.checkbox("Selected Facilities", value=True, key="pd_show_facilities")
                    viz_config["show_assignments"] = st.checkbox("Assignments", value=True, key="pd_show_assignments")
                    
                    # Service radius toggle
                    try:
                        current_problem = (st.session_state.problem_state["problem_type"] or "").lower()
                        params = st.session_state.problem_state.get("parameters", {})
                        sol = st.session_state.problem_state.get("solution", {}) or {}
                        metrics = sol.get("metrics", {})
                        service_radius = params.get("service_radius") or metrics.get("service_radius")
                        
                        if service_radius is not None and current_problem in ["mclp", "lscp"]:
                             viz_config["show_service_areas"] = st.checkbox("Service Radius", value=True, key="pd_show_service_areas")
                    except Exception:
                        pass

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
            with st.container():
                legend_html = st.session_state.pydeck_visualizer.generate_legend_html(
                    problem_type=st.session_state.problem_state["problem_type"],
                    has_solution=solution is not None,
                    parameters=parameters or {},
                    constraints=st.session_state.problem_state.get("constraints", {}) or {},
                    solution=solution or {},
                )
                deck_html = deck.to_html(as_string=True)
                # Ensure body is the positioning context for the legend overlay
                deck_html = deck_html.replace(
                    "<body>",
                    '<body style="margin:0;padding:0;position:relative;">',
                    1,
                )
                # Inject the legend overlay directly inside the deck HTML before </body>
                combined_html = deck_html.replace(
                    "</body>",
                    f"{legend_html}</body>",
                    1,
                )
                components.html(combined_html, height=500)
        else:
            if viz_config is None:
                viz_config = {}
                
            # For Folium, ensure service areas are shown if applicable (LayerControl handles visibility)
            try:
                current_problem = (st.session_state.problem_state["problem_type"] or "").lower()
                params = st.session_state.problem_state.get("parameters", {})
                sol = st.session_state.problem_state.get("solution", {}) or {}
                metrics = sol.get("metrics", {})
                service_radius = params.get("service_radius") or metrics.get("service_radius")
                
                if service_radius is not None and current_problem in ["mclp", "lscp"]:
                     # Checkbox for initial inclusion, though LayerControl can toggle
                     # Placing it here to match previous behavior for Folium
                     viz_config["show_service_areas"] = st.checkbox("Show service radius", value=True, key="folium_show_service_areas")
            except Exception:
                pass

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
                
                # Add sample values for LLM context (first non-null value per column)
                sample_values = {}
                column_stats = {}
                for col in gdf.columns:
                    if col.lower() in ['geometry', 'shape']:
                        continue
                    try:
                        non_null = gdf[col].dropna()
                        if len(non_null) > 0:
                            sample_values[col] = non_null.iloc[0]
                            # Add stats for numeric columns
                            if gdf[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                                column_stats[col] = {
                                    'mean': float(gdf[col].mean()),
                                    'max': float(gdf[col].max())
                                }
                    except Exception:
                        pass
                
                data_summary[name] = {
                    "num_features": len(gdf),
                    "geometry_type": gdf.geometry.type.unique()[0] if len(gdf) > 0 else "Unknown",
                    "columns": [c for c in gdf.columns if c != 'geometry'],
                    "dtypes": dtypes,
                    "bounds": gdf.total_bounds.tolist() if len(gdf) > 0 else [],
                    "capacity_columns": capacity_cols,
                    "cost_columns": cost_cols,
                    "demand_columns": demand_cols,
                    "sample_values": sample_values,
                    "column_stats": column_stats
                }
        
        # Propagate generated-sites settings into problem_state for ADK tools
        st.session_state.problem_state["_generated_sites_count"] = st.session_state.get("generated_sites_count", 100)
        st.session_state.problem_state["_generated_sites_seed"] = st.session_state.get("generated_sites_seed", None)

        # Human-readable labels for ADK tool names shown in the status panel
        _TOOL_LABELS = {
            "fetch_city_data": "Fetching geographic data…",
            "stage_optimization": "Staging optimization parameters…",
            "confirm_optimization": "Running solver…",
            "get_data_status": "Checking data status…",
        }

        # Call agent — use st.status() so users see what's happening
        with st.status("Thinking…", expanded=True) as _status:
            try:
                result = st.session_state.conversation_manager.chat(
                    user_message=prompt,
                    conversation_history=st.session_state.messages[:-1],
                    problem_state=st.session_state.problem_state,
                    uploaded_data_summary=data_summary,
                )
            except Exception as e:
                st.error(f"Error communicating with AI: {e}")
                result = {
                    "response": f"I encountered an error: {str(e)}",
                    "actions": [],
                    "updated_state": st.session_state.problem_state,
                    "tool_calls": [],
                }
            tool_calls = result.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    _status.write(f"✓ {_TOOL_LABELS.get(tc, tc.replace('_', ' ').title())}")
            _status.update(
                label="Done" if tool_calls else "Response ready",
                state="complete",
                expanded=False,
            )

        # Update state
        st.session_state.problem_state = result["updated_state"]

        # Toast notifications for key background actions
        if "fetch_city_data" in tool_calls:
            st.toast("Geographic data loaded — check the map!", icon="🗺️")
        if "confirm_optimization" in tool_calls:
            st.toast("Optimization complete — solution on map!", icon="✅")

        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"]
        })

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

