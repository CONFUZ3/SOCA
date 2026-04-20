# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SOCA (Spatial Optimization Conversational Agent) is a Streamlit web app that lets users describe facility location problems in natural language and solve them with mixed-integer programming. Gemini (via Google ADK) serves as the conversational AI; PuLP (or Gurobi) runs the optimization.

## Commands

```bash
# Run the app
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=solvers --cov=utils --cov=agent

# Run a single test file
pytest tests/test_solvers.py -v

# Run a single test
pytest tests/test_solvers.py::TestPMedianSolver::test_basic -v
```

**API key setup** — create `.env` with `GEMINI_API_KEY=...` or set `GEMINI_API_KEY` in `.streamlit/secrets.toml`.

**Solver**: Gurobi is preferred (`gurobipy` optional) and falls back to PuLP automatically. Time limit and MIP gap are in `config/settings.py` (`SOLVER_TIME_LIMIT`, `MIP_GAP`).

## Architecture

### Request / Response Flow

```
User chat input (app.py)
  → SOCAAgent.chat()                     # agent/soca_agent.py
      → Google ADK Runner invokes Gemini with structured tools
      → Gemini calls ADK tools (function calling, not regex parsing):
          • fetch_city_data    → DataFetcher (utils/data_fetcher.py)
          • stage_optimization → queues optimize action, returns confirmation prompt
          • confirm_optimization → solver via ProblemRegistry
          • get_data_status    → inspects current problem_state
      → state_bridge syncs ADK session ↔ st.session_state.problem_state
  → app.py handles returned actions, updates st.session_state.problem_state
  → render_map_fragment() re-renders PyDeck or Folium map (fragment = no full rerun)
```

The old `ConversationManager` (regex-based JSON parsing) has been replaced by the ADK agent. `agent/conversation_manager.py` may still exist but is no longer the primary path.

### ADK Agent & Tools

**`agent/soca_agent.py`** — `SOCAAgent`
- Wraps Google ADK `LlmAgent` + `Runner`; stateless Gemini calls with full history each turn.
- Structured function calling replaces regex parsing of Gemini output.
- Confirmation gate: `stage_optimization` queues the action; `confirm_optimization` fires it only after explicit user approval.

**`agent/adk_prompts.py`** — builds the ADK system instruction with problem descriptions, tool decision rules, and example workflows.

**`agent/tools/`** — ADK tool implementations:
- `fetch_tools.py` — `fetch_city_data`: boundaries (Overture → Nominatim fallback), population (HDX), POIs (Overture/Overpass).
- `optimize_tools.py` — `stage_optimization` / `confirm_optimization`.
- `status_tools.py` — `get_data_status`.
- `state_bridge.py` — thread-local bridge so ADK tools can read/write `st.session_state` (Streamlit is not thread-safe; the bridge is set before each `Runner.run()` call).

### Data Fetching Pipeline

**`utils/data_fetcher.py`** — `DataFetcher` (rewritten)
- Primary: Overture Maps API for boundaries and POIs.
- Fallback: Nominatim/Photon for boundaries; Overpass for POIs.
- Population: HDX API for population grids; falls back to synthetic grid.
- Per-step error isolation — partial fetches are usable; callers iterate steps independently.
- Exponential backoff retry (1 s → 2 s → 4 s); Nominatim enforces 1-second rate-limit (ToS).

**`utils/geocoder.py`** — `GeocodeCandidate`
- Photon (primary) → Nominatim (fallback) for place-name disambiguation.
- Returns OSM IDs for precise boundary fetching; used by `fetch_city_data` before calling `DataFetcher`.

**`utils/scale_classifier.py`**
- Maps geographic scale (country / region / city / neighbourhood) → OSM `admin_level`.
- Computes synthetic demand-point count from boundary area: `sqrt(area_km2) * 8`, clamped 50–2000.
- Infers scale heuristically from location strings and soft-validates fetched boundary matches.

**`utils/network_manager.py`**
- Lazy-fetches and caches OSMnx road-network graphs (LRU cap = 3 in session state).
- Road-network distance is the **default** `distance_metric` for every solver.
  Graph fetches are wrapped in `utils.activity_log.timed` so the sidebar shows
  `network.fetch / OpenStreetMap` events.
- `launch_prefetch_thread(network_manager, aoi_gdf, session_state)` spawns a
  daemon thread that warms the road-graph cache as soon as the user confirms
  an AOI. Status is written to `st.session_state["_network_status"]`
  (`"fetching" | "ready" | "failed"`) and surfaced in the sidebar with a
  `Refresh road network` button.
- Solvers accept an optional `network_graph` parameter for road-based shortest-path distances instead of Euclidean/haversine.

**`utils/activity_log.py`**
- Structured event bus for user-visible API transparency: stage, status (✓ / … / • / ✗), source attribution, duration.
- Ring buffer (max 50 events); auto-expands in the UI on error.
- Events written by `DataFetcher` and ADK tools; rendered in the sidebar.

**`utils/aoi_selector.py`**
- Leaflet/Folium interactive AOI selector embedded in Streamlit: place-name autocomplete, polygon drawing/editing, basemap toggle (CartoDB / Esri Satellite).

### Solvers

**`solvers/`** — one file per problem type (P-Median, P-Center, MCLP, LSCP).
- All inherit `SpatialOptimizationProblem` (abstract base in `base_solver.py`).
- Each implements: `get_metadata()`, `get_conversation_prompts()`, `get_required_data()`, `validate_parameters()`, `solve()`, `explain_solution()`, `get_visualization_config()`.
- `solve()` returns `{status, objective_value, selected_facilities, assignments, metrics, solution_time}`.
- All four solvers now accept an optional `network_graph` parameter for network distance.
- The default `distance_metric` on every solver is `"network"` (OSM road-network
  shortest path). If the graph is unavailable at solve time,
  `agent/tools/optimize_tools.confirm_optimization` auto-falls back to geodesic
  and attaches a human-readable message to the returned `warnings` list. Pass
  `strict_network=True` to `stage_optimization` when the caller requires road
  distance and wants a hard error on fetch failure instead of the fallback.
- `solvers/registry.py` exports the singleton `problem_registry`; solvers are auto-registered on import.
- `utils/heuristics/genetic_solver.py` provides a genetic algorithm fallback.

### Other Utilities

**`utils/data_processor.py`** — loads GeoJSON, Shapefile (zip), CSV; `identify_data_type()` classifies a GeoDataFrame as `demand_points` or `candidate_sites`; `generate_candidate_sites()` creates random candidates within extent.

**`utils/pydeck_visualizer.py`** / **`utils/visualizer.py`** — PyDeck (WebGL, default) and Folium renderers. Map is rendered inside `@st.fragment render_map_fragment()` to avoid full page reruns.

**`config/settings.py`** — global constants including `ADK_APP_NAME`, `ADK_MAX_TOOL_CALLS_PER_TURN`, solver timeouts, CRS defaults, file limits, and `DATA_DIR` / `TEMP_DIR` / `TEST_DATA_DIR` paths.

### Problem State

`st.session_state.problem_state` is the central shared state threaded through every agent call:

```python
{
    "problem_type": str | None,       # e.g. "p-median", "mclp"
    "parameters": dict,               # n_facilities, service_radius, variant, …
    "constraints": dict,
    "data": dict[str, GeoDataFrame],  # keyed by filename / auto-generated key
    "solution": dict | None,
    "solution_history": list[dict],
    "pending_action": dict | None,    # queued optimize action awaiting confirmation
    "parameters_confirmed": bool,
}
```

Dataset keys follow conventions `app.py` uses to classify role: `boundary_*`, `demand_*`, `*_facilities_*`, `generated_candidates`.

The ADK `state_bridge` mirrors relevant fields into the ADK session context so Gemini has awareness of loaded data and current parameters without re-reading Streamlit state directly.

### Adding a New Solver

1. Create `solvers/my_solver.py` inheriting `SpatialOptimizationProblem`.
2. Implement all abstract methods; `short_name` in `get_metadata()` is the registry key.
3. Register it in `solvers/registry.py` inside `_register_default_problems()`.

### Adding a New ADK Tool

1. Define the function in `agent/tools/` with a typed signature (ADK derives the JSON schema from type hints and docstring).
2. Access `problem_state` via `state_bridge.get_state()` — never import `st` directly inside a tool.
3. Register the function in `SOCAAgent.__init__()` as part of the `tools=` list passed to `LlmAgent`.
