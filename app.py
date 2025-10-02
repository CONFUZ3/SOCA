import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from pathlib import Path
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports from our modules
from agent.conversation_manager import ConversationManager
from solvers.registry import problem_registry
from utils.data_processor import DataProcessor
from utils.visualizer import MapVisualizer
from utils.export_handler import ExportHandler
from config.settings import settings

# Page config
st.set_page_config(
    page_title="Spatial Optimization Agent",
    page_icon="🗺️",
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
            "content": """Welcome to the Spatial Optimization Conversational Agent! 🗺️

I'm here to help you solve facility location problems using state-of-the-art optimization techniques powered by Google Gemini.

**To get started:**
1. Upload your geospatial data (demand points and candidate sites) using the sidebar
2. Describe your optimization problem in natural language
3. I'll guide you through the process and help you find the optimal solution

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
    
    if "conversation_manager" not in st.session_state:
        # Get API key
        api_key = None
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            st.error("⚠️ GEMINI_API_KEY not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
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
    st.title("🗺️ Spatial Optimization")
    st.markdown("### Academic Research Tool")
    
    st.divider()
    
    # File upload section
    st.subheader("📁 Upload Data")
    st.markdown("Upload geospatial data files (GeoJSON, Shapefile, CSV)")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["geojson", "json", "csv", "shp", "zip"],
        accept_multiple_files=True,
        help="Upload demand points, candidate sites, or boundary files",
        key="file_uploader"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.problem_state["data"]:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        gdf = st.session_state.data_processor.load_file(file)
                        gdf = st.session_state.data_processor.preprocess_data(gdf)
                        st.session_state.problem_state["data"][file.name] = gdf
                        st.success(f"✓ Loaded {file.name}: {len(gdf)} features")
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {str(e)}")
    
    # Display loaded data summary
    if st.session_state.problem_state["data"]:
        st.divider()
        st.subheader("📊 Loaded Data")
        for name, gdf in st.session_state.problem_state["data"].items():
            with st.expander(f"📄 {name}"):
                st.write(f"**Features:** {len(gdf)}")
                st.write(f"**Geometry:** {gdf.geometry.type.unique()[0]}")
                st.write(f"**Columns:** {', '.join([c for c in gdf.columns if c != 'geometry'])}")
                st.write(f"**CRS:** {gdf.crs}")
    
    st.divider()
    
    # Problem registry info
    st.subheader("📚 Available Problems")
    problems = problem_registry.list_problems()
    st.write(f"**{len(problems)}** problem types available")
    
    with st.expander("View All Problems"):
        for prob in problems:
            st.markdown(f"**{prob['name']}** (`{prob['short_name']}`)")
            st.caption(prob['description'])
            st.divider()
    
    st.divider()
    
    # Current problem state
    if st.session_state.problem_state["problem_type"]:
        st.subheader("🎯 Current Problem")
        st.info(st.session_state.problem_state["problem_type"])
        
        if st.session_state.problem_state["parameters"]:
            with st.expander("Parameters"):
                st.json(st.session_state.problem_state["parameters"])
        
        if st.session_state.problem_state["constraints"]:
            with st.expander("Constraints"):
                st.json(st.session_state.problem_state["constraints"])
    
    # Clear conversation button
    st.divider()
    if st.button("🔄 Reset Conversation"):
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
st.caption("An academic research tool for facility location problems powered by Claude AI")

# Create two columns: chat and map
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💬 Conversation")
    
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
            data_summary = {
                name: {
                    "num_features": len(gdf),
                    "geometry_type": gdf.geometry.type.unique()[0] if len(gdf) > 0 else "Unknown",
                    "columns": [c for c in gdf.columns if c != 'geometry'],
                    "bounds": gdf.total_bounds.tolist() if len(gdf) > 0 else []
                }
                for name, gdf in st.session_state.problem_state["data"].items()
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
            for action in result["actions"]:
                if action["action"] == "optimize":
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
                                elif "demand_points" not in data_dict:
                                    # Assume first dataset is demand
                                    data_dict["demand_points"] = gdf
                                elif "candidate_sites" not in data_dict:
                                    # Assume second dataset is candidates
                                    data_dict["candidate_sites"] = gdf
                            
                            # Add default weights if needed
                            if "demand_points" in data_dict:
                                data_dict["demand_points"] = data_processor.add_default_weights(data_dict["demand_points"])
                            
                            # Validate required data
                            required_data = problem_solver.get_required_data()
                            missing_data = [k for k, v in required_data.items() if v.get('required') and k not in data_dict]
                            
                            if missing_data:
                                error_msg = f"Missing required data: {', '.join(missing_data)}"
                                st.error(error_msg)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"❌ {error_msg}. Please upload the required data files."
                                })
                                st.rerun()
                                continue
                            
                            # Solve
                            solution = problem_solver.solve(
                                data=data_dict,
                                parameters=action.get("parameters", {}),
                                constraints=action.get("constraints", {}),
                                distance_metric=action.get("distance_metric", "euclidean")
                            )
                            
                            st.session_state.problem_state["solution"] = solution
                            st.session_state.problem_state["solution_history"].append(solution)
                            
                            # Generate explanation
                            if solution.get('status') != 'error':
                                explanation = problem_solver.explain_solution(
                                    solution=solution,
                                    data=data_dict,
                                    detail_level="standard"
                                )
                                
                                # Add explanation to chat
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"✅ **Optimization Complete!**\n\n{explanation}"
                                })
                                
                                st.success("Optimization completed! Check the map for results.")
                            else:
                                error_msg = f"❌ Optimization failed: {solution.get('error', 'Unknown error')}"
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": error_msg
                                })
                                st.error(error_msg)
                        
                        except Exception as e:
                            error_msg = f"❌ Optimization error: {str(e)}"
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                            st.error(error_msg)
                            logger.error(f"Optimization error: {e}", exc_info=True)
        
        st.rerun()

with col2:
    st.subheader("🗺️ Visualization")
    
    # Map display
    if st.session_state.problem_state["data"] or st.session_state.problem_state["solution"]:
        try:
            # Get visualization config if problem type is known
            viz_config = None
            if st.session_state.problem_state["problem_type"]:
                problem_solver = problem_registry.get_problem(st.session_state.problem_state["problem_type"])
                if problem_solver:
                    viz_config = problem_solver.get_visualization_config()
            
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
                    # Assume first dataset is demand
                    mapped_data["demand_points"] = gdf
                elif "candidate_sites" not in mapped_data:
                    # Assume second dataset is candidates
                    mapped_data["candidate_sites"] = gdf
            
            # Create map
            map_obj = st.session_state.map_visualizer.create_map(
                data=mapped_data,
                solution=st.session_state.problem_state["solution"],
                problem_type=st.session_state.problem_state["problem_type"],
                viz_config=viz_config,
                parameters=st.session_state.problem_state.get("parameters", {}),
                constraints=st.session_state.problem_state.get("constraints", {})
            )
            
            # Display map
            st_folium(map_obj, width=700, height=500, key="map")
        
        except Exception as e:
            st.error(f"Error creating map: {e}")
            logger.error(f"Map error: {e}", exc_info=True)
        
        # Metrics dashboard
        if st.session_state.problem_state["solution"]:
            st.divider()
            st.subheader("📈 Solution Metrics")
            
            solution = st.session_state.problem_state["solution"]
            
            if solution.get('status') in ['optimal', 'feasible']:
                metrics = solution.get("metrics", {})
                
                # Display key metrics in columns
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    obj_val = solution.get('objective_value', 0)
                    if obj_val is not None:
                        st.metric(
                            "Objective Value",
                            f"{obj_val:.2f}"
                        )
                
                with metric_cols[1]:
                    st.metric(
                        "Solution Status",
                        solution.get('status', 'Unknown').title()
                    )
                
                with metric_cols[2]:
                    st.metric(
                        "Solution Time",
                        f"{solution.get('solution_time', 0):.2f}s"
                    )
                
                # Additional metrics
                with st.expander("📊 Detailed Metrics"):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            st.write(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # Export options
                st.divider()
                st.subheader("💾 Export Solution")
                
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    if st.button("📄 Export GeoJSON"):
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
                    if st.button("📊 Export CSV"):
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
                    if st.button("📝 Generate PDF Report"):
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
            else:
                st.warning(f"Solution status: {solution.get('status', 'Unknown')}")
                if solution.get('error'):
                    st.error(solution['error'])
    
    else:
        st.info("👆 Upload data or start a conversation to see visualizations")
        
        # Show example
        with st.expander("📖 Quick Start Guide"):
            st.markdown("""
            **Step 1: Upload Data**
            - Upload demand points (e.g., population centers)
            - Upload candidate sites (e.g., potential facility locations)
            
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
    st.caption("🔬 Academic Research Tool")
with col_f2:
    st.caption(f"🤖 Powered by Google Gemini")
with col_f3:
    st.caption(f"📚 {len(problems)} Problem Types Available")

