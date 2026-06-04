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

**Solver**: Gurobi is preferred (`gurobipy` optional) and falls back to PuLP automatically. Time limit and MIP gap are in `config/settings.py` (`SOLVER_MIP_TIME_LIMIT`, `MIP_GAP`).

**CORS**: `SOCA_CORS_ORIGINS` env var (comma-separated) configures FastAPI allowed origins; defaults to `http://localhost:3000`. The `X-Session-Id` header is exposed in CORS config.

## Architecture

### Request / Response Flow (FastAPI + Next.js)

```
Next.js UI (frontend/app/page.tsx)
  → REST/SSE calls to FastAPI (backend/api/*)
      • POST /api/session              → SessionStore allocates a session (cookie: soca_session)
      • POST /api/chat/stream          → SSE: agent tokens + tool events
      • GET  /api/events/stream        → SSE: activity-log + network-status events
      • POST /api/aoi/resolve, /confirm → AOI boundary fetch + lock
      • GET  /api/aoi/suggest          → geocode autocomplete
      • GET  /api/network/status       → road-graph prefetch status
      • POST /api/data/upload          → file upload
      • GET  /api/map/state            → MapLibre layer payloads + solution summary
      • GET  /api/export/{geojson,csv,pdf} → solution export
  → backend.api.chat.chat_stream() drives SOCAAgent.chat_stream()  # agent/soca_agent.py
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

`backend/services/session_store.py` replaces `st.session_state` for the API path: each session owns its own `problem_state` dict, keyed by a `soca_session` cookie (24-char URL-safe token, 12h TTL). `backend/services/event_bus.py` is the SSE pub/sub for activity-log events — `bind_session(session_id)` sets thread-local routing so tool callbacks fan to the right subscriber queue.

The legacy Streamlit `app.py` still works against the same agent/solvers/utils, using `st.session_state` directly. The old `ConversationManager` (regex-based JSON parsing) is fully superseded by the ADK agent; `agent/conversation_manager.py` is dead code.

### ADK Agent & Tools

**`agent/soca_agent.py`** — `SOCAAgent(api_key, problem_registry)`
- Wraps Google ADK `LlmAgent` + `Runner`; stateless Gemini calls with full history each turn.
- `async chat_stream(user_message, conversation_history, problem_state, uploaded_data_summary)` → async generator yielding structured events: `tool_call_start`, `tool_call_result`, `token`, `final`, `error`.
- Confirmation gate: `stage_optimization` queues the action; `confirm_optimization` fires it only after explicit user approval.

**`agent/adk_prompts.py`** — builds the ADK system instruction with problem descriptions, tool decision rules, and example workflows.

**`agent/tools/`** — ADK tool implementations (imported individually in `soca_agent.py`; `__init__.py` is a marker):
- `fetch_tools.py` — `fetch_city_data`: boundaries (Overture → Nominatim fallback), population (HDX), POIs (Overture/Overpass).
- `optimize_tools.py` — `stage_optimization` / `confirm_optimization`. Confirmation also writes a reproducibility record via `utils/repro_logger.py`, attaches `equity_metrics`, and builds an `analysis_facts` block (via `utils/solution_report.py`) — a fully unit-labeled facts payload (distance distribution in km, per-facility breakdown with reverse-geocoded place names, located coverage gaps, interpretable equity) that the agent narrates from. The deterministic `solution_summary` template remains as a fallback.
- `status_tools.py` — `get_data_status`.
- `sensitivity_tools.py` — `run_sensitivity_analysis`: drop-one re-optimization over each selected facility, reporting per-facility objective degradation and the `most_critical` facility. Reuses cached road graph + `data_dict`; never re-fetches.
- `state_bridge.py` — thread-local bridge so ADK tools can read/write the active session's `problem_state` (works for both Streamlit `st.session_state` and the FastAPI `SessionStore`); call `bind_session()` before each `Runner.run()`.

### Data Fetching Pipeline

**`utils/fetchers/`** — modular fetcher package (replaces the monolithic `data_fetcher.py`, which is now a thin re-export shim for backward compat).
- `facade.py` — `DataFetcher` class; public API: `fetch_boundaries(location, admin_level, scale, hint)` / `fetch_pois(boundary_gdf, category)` / `fetch_population(boundary_gdf, n_points, random_seed)`.
- `boundaries.py` — Overture Maps (primary) → Nominatim/Photon (fallback).
- `pois.py` — Overture (primary) → `pois_overpass.py` (fallback; fills sparse-coverage regions).
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
  Graph fetches are wrapped in `utils.activity_log.timed` so the sidebar shows `network.fetch / OpenStreetMap` events.
- `launch_prefetch_thread(network_manager, aoi_gdf, session_state)` spawns a daemon thread that warms the road-graph cache as soon as the user confirms an AOI. Status is written to `_network_status` (`"fetching" | "ready" | "failed"`) in session state and surfaced in the frontend.
- Solvers accept an optional `network_graph` parameter for road-based shortest-path distances instead of Euclidean/haversine.

**`utils/distance_calculator.py`**
- Metrics: geodesic (haversine via `pyproj.Geod`), Euclidean (2D/3D), network (delegates to `network_manager`).
- LRU cache (max 10 matrices) keyed by coords + metric + CRS; unit conversion for m/km/mi/ft/yd/nm.

**`utils/activity_log.py`**
- Structured event bus for user-visible API transparency: stage, status (✓ / … / • / ✗), source attribution, duration.
- Ring buffer (max 50 events); auto-expands in the UI on error.
- Events written by `DataFetcher` and ADK tools; bridged to the FastAPI SSE stream via `backend/services/event_bus.py`.

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
- All four solvers accept an optional `network_graph` parameter for road-network shortest-path distances.
- The default `distance_metric` on every solver is `"network"` (OSM road-network shortest path). If the graph is unavailable at solve time, `confirm_optimization` auto-falls back to geodesic and attaches a human-readable message to the returned `warnings` list. Pass `strict_network=True` to `stage_optimization` to require road distance and get a hard error on fetch failure.
- `solvers/registry.py` exports the singleton `problem_registry`; registered solvers: `PMedianSolver` ("p-median"), `PCenterSolver` ("p-center"), `MCLPSolver` ("mclp"), `LSCPSolver` ("lscp").
- `utils/heuristics/genetic_solver.py` provides a genetic algorithm fallback.

### Other Utilities

**`utils/data_processor.py`** — loads GeoJSON, Shapefile (zip), CSV; `identify_data_type()` classifies a GeoDataFrame as `demand_points` or `candidate_sites`; `generate_candidate_sites()` creates random candidates within extent.

**`utils/pydeck_visualizer.py`** / **`utils/visualizer.py`** — PyDeck (WebGL, default) and Folium renderers. Map is rendered inside `@st.fragment render_map_fragment()` to avoid full page reruns.

**`config/settings.py`** — global constants:
- Solver: `SOLVER_MIP_TIME_LIMIT` (120s), `SOLVER_GA_TIME_LIMIT` (120s), `SOLVER_WALL_CLOCK_TIMEOUT` (420s), `MIP_GAP` (0.01), `MIP_MODEL_SIZE_LIMIT` (300 000).
- Gurobi tuning: `GUROBI_PRESOLVE` (2), `GUROBI_CUTS` (2), `GUROBI_HEURISTICS` (0.05), `GUROBI_MIP_FOCUS` (1), `GUROBI_THREADS` (0 = auto).
- Network: `NETWORK_FETCH_TIMEOUT` (90s), `NETWORK_DIJKSTRA_BUDGET_SECONDS` (120), `NETWORK_FETCH_MAX_AREA_KM2` (10 000), `NETWORK_AUTO_EUCLIDEAN_AREA_KM2` (2 000), `DEFAULT_DRIVE_SPEED_KMH` (30).
- Candidates: `MAX_CANDIDATE_SITES` (500), `CANDIDATE_THINNING_MIN_DIST_M` (200).
- CRS: `CRS_STANDARD` (EPSG:4326), `CRS_PROJECTED` (EPSG:3857).
- Upload: `MAX_UPLOAD_SIZE_MB` (50), `ALLOWED_EXTENSIONS` (.geojson, .json, .csv, .shp, .zip).
- ADK: `ADK_APP_NAME = "soca"`.
- Paths: `BASE_DIR`, `DATA_DIR`, `TEMP_DIR`, `DOCS_DIR`, `TEST_DATA_DIR`, `RUNS_DIR`.
- Reproducibility: `RANDOM_SEED` (42).

### Backend (FastAPI)

**`backend/main.py`** — `app = FastAPI(...)`; CORS via `SOCA_CORS_ORIGINS` (exposes `X-Session-Id`); warms `SessionStore` + `EventBus` singletons in lifespan; mounts all routers under `/api/*`.

**`backend/api/`** — one router per resource:

| Module | Prefix | Key routes |
|--------|--------|------------|
| `session.py` | `/api/session` | POST (create/refresh), GET (snapshot), DELETE (reset) |
| `problems.py` | `/api/problems` | GET (list registered solvers + metadata) |
| `aoi.py` | `/api/aoi` | GET `/suggest` (geocode), POST `/resolve` (boundary fetch), POST `/confirm` (AOI lock + network prefetch) |
| `network.py` | `/api/network` | GET `/status`, POST `/refresh` (manual prefetch) |
| `events.py` | `/api/events` | GET `/stream` (SSE activity-log + network-status, heartbeat every 15 s) |
| `chat.py` | `/api/chat` | POST `/stream` (SSE agent tokens + tool calls), GET `/history` |
| `data.py` | `/api/data` | GET (list datasets), POST `/upload`, DELETE `/{name}`, PATCH `/{name}/filter`, GET `/{name}.geojson` |
| `map.py` | `/api/map` | GET `/state` (MapLibre layers + solution summary) |
| `export.py` | `/api/export` | GET `/geojson`, `/csv`, `/pdf` |

**`backend/deps.py`** — session resolution helpers:
- `SESSION_COOKIE = "soca_session"`, `COOKIE_MAX_AGE = 43200` (12 h).
- `resolve_session(request, response, cookie)` — creates/refreshes session, returns `(session_id, record)`; used by most endpoints.
- `require_session(request, cookie)` — fails with 401 if no valid session; used by `/api/events/stream`.
- `get_store()` → `SessionStore`, `get_bus()` → `EventBus`.

**`backend/services/`**:
- `session_store.py` — in-memory per-session state keyed by `soca_session` cookie; `SESSION_TTL_SECONDS = 43200`; garbage-collected via `sweep()`. Each record holds `problem_state` plus internal keys (`messages`, `_activity_log`, `_network_status`, `_network_manager`, `_data_fetcher`, `_soca_agent`, etc.).
- `event_bus.py` — async pub/sub; `bind_session(id)` sets thread-local scope; `subscribe(session_id)` returns an `asyncio.Queue` (max 256); `BusEvent(kind, payload)` serialised as SSE JSON frames.

### Frontend (Next.js 16)

**`frontend/`** — Next.js 16.2.4 + Tailwind, hand-crafted design system. Single-container deploy via the root `Dockerfile`.
- `app/page.tsx` — renders `<Workspace />`; all layout logic is in components.
- `components/{chat,map,sidebar,aoi,layout,ui}` — feature components; `map/` is MapLibre GL; `ui/` is a shadcn-style primitive library (button, input, dialog, tab, card, chip, kbd, popover, tooltip).
- `hooks/` — `use-session.ts` (session init + reset), `use-chat.ts` (SSE chat stream), `use-events-stream.ts` (SSE activity log → store), `use-map-state.ts` (TanStack Query to `/api/map/state`).
- `lib/store.ts` — Zustand store; state: `snapshot`, `ready`, `items` (ChatTurn | ActivityGroup), `network`, `datasets`; activity groups coalesce within a 4 s window (`ACTIVITY_GROUP_WINDOW_MS`).
- `lib/api.ts` — typed fetch helpers (`apiGet`, `apiPost`, `apiDelete`, `apiUpload`); throws `ApiError(status, detail)` on non-ok.
- `lib/sse.ts`, `lib/sources.ts`, `lib/format.ts`, `lib/cn.ts`, `lib/activity-format.ts` — SSE parsing, source labels, formatters.
- `types/index.ts` — shared TypeScript types.

### Problem State

`problem_state` is the central shared state threaded through every agent call. It lives in `st.session_state["problem_state"]` for Streamlit and in `SessionStore` for the FastAPI path. Canonical shape (from `_fresh_record()`):

```python
{
    "problem_type": str | None,       # e.g. "p-median", "mclp"
    "parameters": dict,               # n_facilities, service_radius, variant, …
    "constraints": dict,
    "data": dict[str, GeoDataFrame],  # keyed by filename / auto-generated key
    "solution": dict | None,
    "solution_history": list[dict],
    "aoi": GeoDataFrame | None,       # confirmed AOI boundary
    "aoi_confirmed": bool,
}
```

Dataset keys follow conventions used to classify role: `boundary_*`, `demand_*`, `*_facilities_*`, `generated_candidates`.

The ADK `state_bridge` mirrors relevant fields into the ADK session context so Gemini has awareness of loaded data and current parameters without reading session state directly.

### Deployment

**`deploy/`** — production deployment config:
- `nginx.conf` — reverse proxy routing frontend + backend.
- `supervisord.conf` — process manager (uvicorn + next.js).
- `README.md` — deployment guide.
- Root `Dockerfile` — single-container build serving both processes.

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
2. Resolve the active session via `backend.deps.resolve_session` (cookie `soca_session`); mutate `session.problem_state` instead of `st.session_state`.
3. Include the router in `backend/main.py` via `app.include_router(...)`.
4. For activity-log surfacing, publish via `backend.services.event_bus` so the existing `/api/events/stream` SSE pushes it to the UI.
