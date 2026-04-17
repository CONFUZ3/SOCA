# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SOCA (Spatial Optimization Conversational Agent) is a Streamlit web app that lets users describe facility location problems in natural language and solve them with mixed-integer programming. Gemini serves as the conversational AI; PuLP (or Gurobi) runs the optimization.

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
  → ConversationManager.chat()          # agent/conversation_manager.py
      → builds system prompt            # agent/prompts.py
      → calls Gemini API (stateless; full history sent each time)
      → _parse_response(): extracts JSON action blocks from Gemini text
          • action == "fetch_data"  → DataFetcher (utils/data_fetcher.py)
          • action == "optimize"    → solver via ProblemRegistry
  → app.py handles returned actions, updates st.session_state.problem_state
  → render_map_fragment() re-renders PyDeck or Folium map (fragment = no full rerun)
```

### Key Components

**`agent/conversation_manager.py`** — `ConversationManager`
- Stateless Gemini calls; full conversation history is rebuilt and sent every request.
- `_parse_response()` looks for JSON blocks in Gemini output to detect `optimize` or `fetch_data` actions.
- Requires explicit user confirmation ("yes", "proceed", etc.) before dispatching `optimize`.
- `_extract_state_updates()` uses regex heuristics to pull `n_facilities`, `service_radius`, `budget`, `variant`, etc. from raw conversation text and merge them into `problem_state`.
- `_normalize_action()` maps synonym parameter names and prevents spurious variant inference.

**`solvers/`** — one file per problem type
- All solvers inherit `SpatialOptimizationProblem` (abstract base in `base_solver.py`).
- Each solver implements: `get_metadata()`, `get_conversation_prompts()`, `get_required_data()`, `validate_parameters()`, `solve()`, `explain_solution()`, `get_visualization_config()`.
- `solve()` returns a dict with `status`, `objective_value`, `selected_facilities`, `assignments`, `metrics`, `solution_time`.
- `solvers/registry.py` exports the singleton `problem_registry`; all four solvers (P-Median, P-Center, MCLP, LSCP) are auto-registered on import.
- `utils/heuristics/genetic_solver.py` provides a genetic algorithm fallback.

**`utils/data_fetcher.py`** — `DataFetcher`
- Fetches boundaries (Nominatim → Overpass fallback), synthetic population grids, and POIs (Overpass) from public APIs without requiring user uploads.
- Raises `DataFetchError` / subclasses per step; callers iterate steps independently so partial failures don't abort the whole fetch.
- Nominatim requests enforce a 1-second rate-limit delay (ToS).

**`utils/data_processor.py`** — `DataProcessor`
- Loads GeoJSON, Shapefile (zip), and CSV with lat/lon or x/y columns.
- `identify_data_type()` heuristically classifies a GeoDataFrame as `demand_points` or `candidate_sites`.
- `generate_candidate_sites()` creates random point candidates within a demand/boundary extent when no candidate file is uploaded.

**`utils/pydeck_visualizer.py`** and **`utils/visualizer.py`**
- PyDeck (WebGL, default) and Folium (Leaflet, classic) renderers respectively.
- Map is rendered inside `@st.fragment render_map_fragment()` in `app.py` to avoid triggering a full Streamlit page rerun on map interaction.

**`config/settings.py`** — global constants (model name, solver timeouts, CRS defaults, file limits).

### Problem State

`st.session_state.problem_state` is the central shared state passed into every `ConversationManager.chat()` call:

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

Dataset keys follow naming conventions that `app.py` uses to classify role (boundary vs. demand vs. candidate): `boundary_*`, `demand_*`, `*_facilities_*`, `generated_candidates`.

### Adding a New Solver

1. Create `solvers/my_solver.py` inheriting `SpatialOptimizationProblem`.
2. Implement all abstract methods; `short_name` in `get_metadata()` is the registry key.
3. Register it in `solvers/registry.py` inside `_register_default_problems()`.
