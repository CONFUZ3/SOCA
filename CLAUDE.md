# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SOCA (Spatial Optimization Conversational Agent) lets users describe facility location problems in natural language and solve them with mixed-integer programming. Gemini (via Google ADK) is the conversational AI; PuLP (or Gurobi) runs the optimization.

The product is now a **FastAPI backend + Next.js frontend** (`backend/` + `frontend/`). The original Streamlit `app.py` still exists for legacy/local use, but the primary path is the API + React UI.

## Commands

```bash
# Backend (FastAPI) — primary
uvicorn backend.main:app --reload --port 8000

# Frontend (Next.js 16) — primary UI at http://localhost:3000
cd frontend && npm install && npm run dev

# Legacy Streamlit app
streamlit run app.py

# Install Python deps
pip install -r requirements.txt

# Tests
pytest tests/ -v
pytest tests/ -v --cov=solvers --cov=utils --cov=agent
pytest tests/test_solvers.py::TestPMedianSolver::test_basic -v
```

**API key setup** — create `.env` with `GEMINI_API_KEY=...` (loaded by `backend/main.py` via `python-dotenv`; Streamlit also reads `.streamlit/secrets.toml`).

**Solver**: Gurobi is preferred (`gurobipy` optional) and falls back to PuLP automatically. Time limit and MIP gap are in `config/settings.py` (`SOLVER_TIME_LIMIT`, `MIP_GAP`).

**CORS**: `SOCA_CORS_ORIGINS` env var (comma-separated) configures FastAPI allowed origins; defaults to `http://localhost:3000`.

## Architecture

### Request / Response Flow (FastAPI + Next.js)

```
Next.js UI (frontend/app/page.tsx)
  → REST/SSE calls to FastAPI (backend/api/*)
      • POST /api/session              → SessionStore allocates a session
      • POST /api/chat/stream          → SSE: agent tokens + tool events
      • GET  /api/events/stream        → SSE: activity-log events
      • POST /api/aoi, /api/data, …    → state mutations
      • GET  /api/map                  → MapLibre layer payloads
  → backend.api.chat.chat_stream() drives SOCAAgent.chat()  # agent/soca_agent.py
      → Google ADK Runner invokes Gemini with structured tools
      → Gemini calls ADK tools (function calling, not regex parsing):
          • fetch_city_data           → DataFetcher  (utils/fetchers/)
          • stage_optimization        → queues optimize action
          • confirm_optimization      → solver via ProblemRegistry
          • get_data_status           → inspects problem_state
          • run_sensitivity_analysis  → drop-one re-solve on last solution
      → state_bridge syncs ADK session ↔ SessionStore (per-session problem_state)
  → backend.services.event_bus broadcasts activity-log + tool events over SSE
  → Frontend updates chat, sidebar, and MapLibre layers from streamed events
```

`backend/services/session_store.py` replaces `st.session_state` for the API path: each session owns its own `problem_state` dict. `backend/services/event_bus.py` is the SSE pub/sub for activity-log events.

The legacy Streamlit `app.py` still works against the same agent/solvers/utils, using `st.session_state` directly. The old `ConversationManager` (regex-based JSON parsing) is fully superseded by the ADK agent; `agent/conversation_manager.py` is dead code.

### ADK Agent & Tools

**`agent/soca_agent.py`** — `SOCAAgent`
- Wraps Google ADK `LlmAgent` + `Runner`; stateless Gemini calls with full history each turn.
- Structured function calling replaces regex parsing of Gemini output.
- Confirmation gate: `stage_optimization` queues the action; `confirm_optimization` fires it only after explicit user approval.

**`agent/adk_prompts.py`** — builds the ADK system instruction with problem descriptions, tool decision rules, and example workflows.

**`agent/tools/`** — ADK tool implementations:
- `fetch_tools.py` — `fetch_city_data`: boundaries (Overture → Nominatim fallback), population (HDX), POIs (Overture/Overpass).
- `optimize_tools.py` — `stage_optimization` / `confirm_optimization`. Confirmation also writes a reproducibility record via `utils/repro_logger.py` and attaches `equity_metrics` to the result.
- `status_tools.py` — `get_data_status`.
- `sensitivity_tools.py` — `run_sensitivity_analysis`: drop-one re-optimization over each selected facility, reporting per-facility objective degradation and the `most_critical` facility. Reuses cached road graph + `data_dict`; never re-fetches.
- `state_bridge.py` — thread-local bridge so ADK tools can read/write the active session's `problem_state` (works for both Streamlit `st.session_state` and the FastAPI `SessionStore`); set before each `Runner.run()`.

### Data Fetching Pipeline

**`utils/fetchers/`** — modular fetcher package (replaces the monolithic `data_fetcher.py`, which is now a thin re-export shim for backward compat).
- `facade.py` — `DataFetcher` class; preserves the legacy public API (`fetch_boundaries` / `fetch_pois` / `fetch_population`).
- `boundaries.py` — Overture Maps (primary) → Nominatim/Photon (fallback).
- `pois.py` — Overture (primary) → Overpass (fallback).
- `population.py` — HDX population grids; falls back to synthetic grid.
- `overture_duckdb.py` / `overture_release.py` — DuckDB-based Overture queries against the public S3 release.
- `http.py` — shared `requests.Session` + token-bucket rate limiter; exponential backoff (1 s → 2 s → 4 s); Nominatim 1-req/s ToS enforcement.
- `validation.py`, `errors.py`, `constants.py` — shared helpers and exception types (`DataFetchError`, `GeocodingError`, `PopulationDataError`).
- Per-step error isolation: partial fetches are usable; callers iterate steps independently.

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
- Events written by `DataFetcher` and ADK tools; rendered in the Streamlit sidebar and bridged to the FastAPI SSE stream via `backend/services/event_bus.py`.

**`utils/equity_metrics.py`**
- Post-solve equity metrics (weighted Gini, top-decile share, etc.) computed from the assignment + demand weights and attached to every solver result so the agent's summary always pairs efficiency (objective) with distributional impact.

**`utils/repro_logger.py`**
- Writes a JSON record per `confirm_optimization` to `RUNS_DIR`: inputs, seed, solver+parameters, result. Replay is a documented stub (upstream data — HDX/Overture/OSM — isn't frozen). Seed is `SOCA_RANDOM_SEED` env or `config.settings.RANDOM_SEED` (default 42).

**`utils/aoi_selector.py`**
- Leaflet/Folium interactive AOI selector embedded in Streamlit: place-name autocomplete, polygon drawing/editing, basemap toggle (CartoDB / Esri Satellite). The Next.js UI uses MapLibre GL via `frontend/components/map/` instead.

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

**`config/settings.py`** — global constants including `ADK_APP_NAME`, `ADK_MAX_TOOL_CALLS_PER_TURN`, solver timeouts, CRS defaults, file limits, `RANDOM_SEED`, and `DATA_DIR` / `TEMP_DIR` / `TEST_DATA_DIR` / `RUNS_DIR` / `CACHE_DIR` paths.

### Backend (FastAPI)

**`backend/main.py`** — `app = FastAPI(...)`; CORS via `SOCA_CORS_ORIGINS`; warms `SessionStore` + `EventBus` singletons in lifespan; mounts routers under `/api/*`.

**`backend/api/`** — one router per resource: `session`, `problems`, `aoi`, `network`, `events` (SSE activity log), `chat` (SSE token stream), `data`, `map` (MapLibre payloads), `export`.

**`backend/services/`**:
- `session_store.py` — in-memory per-session `problem_state` keyed by session ID (header `X-Session-Id`); replaces `st.session_state` for the API path.
- `event_bus.py` — async pub/sub fanning activity-log + tool events out to SSE subscribers.

### Frontend (Next.js 16)

**`frontend/`** — Next.js 16.2.4 + Tailwind, hand-crafted design system. Single-container deploy via the root `Dockerfile`.
- `app/` — App Router root (`page.tsx`, `layout.tsx`, `globals.css`).
- `components/{chat,map,sidebar,aoi,layout,ui}` — feature components; `map/` is MapLibre GL.
- `hooks/`, `lib/`, `types/` — shared client utilities and types.

### Problem State

`problem_state` is the central shared state threaded through every agent call. It lives in `st.session_state["problem_state"]` for Streamlit and in `SessionStore` (keyed by session id) for the FastAPI path:

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

Dataset keys follow conventions used to classify role: `boundary_*`, `demand_*`, `*_facilities_*`, `generated_candidates`.

The ADK `state_bridge` mirrors relevant fields into the ADK session context so Gemini has awareness of loaded data and current parameters without reading session state directly.

### Adding a New Solver

1. Create `solvers/my_solver.py` inheriting `SpatialOptimizationProblem`.
2. Implement all abstract methods; `short_name` in `get_metadata()` is the registry key.
3. Register it in `solvers/registry.py` inside `_register_default_problems()`.

### Adding a New ADK Tool

1. Define the function in `agent/tools/` with a typed signature (ADK derives the JSON schema from type hints and docstring).
2. Access `problem_state` via `state_bridge.get_state()` — never import `st` directly inside a tool (the same tool runs under both Streamlit and the FastAPI session store).
3. Register the function in `SOCAAgent.__init__()` as part of the `tools=` list passed to `LlmAgent`.

### Adding a New Backend Endpoint

1. Add a router module in `backend/api/` and export `router = APIRouter(prefix="/api/...")`.
2. Resolve the active session via `backend.deps.get_session` (header `X-Session-Id`); mutate `session.problem_state` instead of `st.session_state`.
3. Include the router in `backend/main.py` via `app.include_router(...)`.
4. For activity-log surfacing, publish via `backend.services.event_bus` so the existing `/api/events/stream` SSE pushes it to the UI.
