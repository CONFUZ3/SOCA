# SOCA System Architecture

**Spatial Optimization Conversational Agent — Server Architecture**

---

## Figure 1 — Layered System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           PRESENTATION LAYER                                     ║
║                        Next.js 16.2.4  (port 3000)                              ║
║                                                                                  ║
║  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  ┌─────────────┐ ║
║  │   Chat Panel    │  │  MapLibre GL     │  │    Sidebar    │  │  AOI Draw   │ ║
║  │  (SSE tokens)   │  │  (layer payloads)│  │ (activity log)│  │  (polygon)  │ ║
║  └────────┬────────┘  └────────┬─────────┘  └──────┬────────┘  └──────┬──────┘ ║
╚═══════════╪════════════════════╪═══════════════════╪══════════════════╪═════════╝
            │  HTTP / SSE        │  HTTP / SSE        │  SSE             │  HTTP
            │  (REST + stream)   │                    │                  │
╔═══════════╪════════════════════╪═══════════════════╪══════════════════╪═════════╗
║           │           APPLICATION GATEWAY LAYER                        │         ║
║           │              FastAPI 0.x  (port 8000)                      │         ║
║           │                                                             │         ║
║  ┌────────▼────────────────────────────────────────────────────────────▼──────┐ ║
║  │                         Router Registry                                     │ ║
║  │                                                                              │ ║
║  │  /api/session   /api/chat/stream   /api/events/stream   /api/aoi            │ ║
║  │  /api/data      /api/map           /api/network          /api/export         │ ║
║  │  /api/problems  /api/health                                                  │ ║
║  └───┬────────┬──────────────────────────────────────────────────────────┬─────┘ ║
║      │        │                                                           │       ║
║  ┌───▼──────┐ │  ┌──────────────────────────────────────────────────┐   │       ║
║  │  Session │ │  │                Service Layer                      │   │       ║
║  │  Deps    │ │  │                                                    │   │       ║
║  │(X-Session│ │  │  ┌──────────────────────┐  ┌───────────────────┐ │   │       ║
║  │   -Id)   │ │  │  │    SessionStore       │  │    EventBus       │ │   │       ║
║  └───┬──────┘ │  │  │  (per-session         │  │  (async pub/sub   │ │   │       ║
║      │        │  │  │   problem_state dict) │  │   SSE fan-out)    │ │   │       ║
║      │        │  │  └──────────┬───────────┘  └────────┬──────────┘ │   │       ║
║      │        │  └─────────────╪────────────────────────╪────────────┘   │       ║
╚══════╪════════╪════════════════╪════════════════════════╪════════════════╪═══════╝
       │        │   state r/w    │                        │ broadcast      │
╔══════╪════════╪════════════════╪════════════════════════╪════════════════╪═══════╗
║      │        │        AGENT / INTELLIGENCE LAYER       │                │       ║
║      │        │                                         │                │       ║
║  ┌───▼────────▼──────────────────────────────────────┐ │                │       ║
║  │                    SOCAAgent                       │ │                │       ║
║  │              (agent/soca_agent.py)                 │ │                │       ║
║  │                                                    │ │                │       ║
║  │  ┌──────────────────┐    ┌────────────────────┐   │ │                │       ║
║  │  │  Google ADK       │    │  StateBridge       │   │ │                │       ║
║  │  │  LlmAgent        │    │  (thread-local      │   │ │                │       ║
║  │  │  + Runner        │    │   context binding)  │   │ │                │       ║
║  │  └────────┬─────────┘    └────────────────────┘   │ │                │       ║
║  │           │ function calls (ADK structured)        │ │                │       ║
║  │  ┌────────▼──────────────────────────────────────┐ │ │                │       ║
║  │  │                  Tool Registry                  │ │ │                │       ║
║  │  │                                                 │ │ │                │       ║
║  │  │  ┌───────────────┐  ┌───────────────────────┐  │ │ │                │       ║
║  │  │  │ fetch_city_   │  │ stage_optimization /  │  │ │ │                │       ║
║  │  │  │ data          │  │ confirm_optimization  │  │ │ │                │       ║
║  │  │  └───────┬───────┘  └──────────┬────────────┘  │ │ │                │       ║
║  │  │          │                      │               │ │ │                │       ║
║  │  │  ┌───────▼───────┐  ┌──────────▼────────────┐  │ │ │                │       ║
║  │  │  │ get_data_     │  │ run_sensitivity_      │  │ │ │                │       ║
║  │  │  │ status        │  │ analysis              │  │ │ │                │       ║
║  │  │  └───────────────┘  └───────────────────────┘  │ │ │                │       ║
║  │  └─────────────────────────────────────────────────┘ │ │                │       ║
║  └──────────────────────────────────────────────────────┘ │                │       ║
║                         │ Gemini API calls                 │ emit events    │       ║
╚═════════════════════════╪════════════════════════════════╪═════════════════╪═══════╝
                          │                                  │                │
╔═════════════════════════╪════════════════════════════════╪═════════════════╪═══════╗
║          COMPUTATION LAYER                                 │                │       ║
║                         │                                  │                │       ║
║  ┌──────────────────────▼──────────────────────────────┐  │                │       ║
║  │                   Solver Registry                    │  │                │       ║
║  │               (solvers/registry.py)                  │  │                │       ║
║  │                                                       │  │                │       ║
║  │  ┌────────────┐ ┌────────────┐ ┌──────┐ ┌────────┐  │  │                │       ║
║  │  │  P-Median  │ │  P-Center  │ │ MCLP │ │  LSCP  │  │  │                │       ║
║  │  │ (minimize  │ │ (minimize  │ │(max  │ │(min    │  │  │                │       ║
║  │  │  avg dist) │ │  max dist) │ │cover)│ │sites)  │  │  │                │       ║
║  │  └────────────┘ └────────────┘ └──────┘ └────────┘  │  │                │       ║
║  │                       │                              │  │                │       ║
║  │  ┌────────────────────▼──────────────────────────┐  │  │                │       ║
║  │  │         Solver Backend Selection               │  │  │                │       ║
║  │  │   Gurobi (preferred)  →  PuLP (fallback)       │  │  │                │       ║
║  │  └───────────────────────────────────────────────┘  │  │                │       ║
║  └──────────────────────────────────────────────────────┘  │                │       ║
║                                                             │                │       ║
║  ┌──────────────────────────────────────────────────────┐  │                │       ║
║  │               Data Fetching Pipeline                  │  │                │       ║
║  │              (utils/fetchers/facade.py)               ├──┘                │       ║
║  │                                                       │                   │       ║
║  │  ┌──────────────┐  ┌────────────────┐  ┌──────────┐  │                   │       ║
║  │  │ boundaries.py│  │   pois.py      │  │population│  │                   │       ║
║  │  │              │  │                │  │  .py     │  │                   │       ║
║  │  │ Overture Maps│  │ Overture Maps  │  │  HDX     │  │                   │       ║
║  │  │ → Nominatim  │  │ → Overpass     │  │ → synth  │  │                   │       ║
║  │  └──────┬───────┘  └───────┬────────┘  └────┬─────┘  │                   │       ║
║  │         │                  │                 │         │                   │       ║
║  │  ┌──────▼──────────────────▼─────────────────▼─────┐  │                   │       ║
║  │  │    http.py — shared Session + token-bucket       │  │                   │       ║
║  │  │    rate limiter + exponential backoff            │  │                   │       ║
║  │  └─────────────────────────────────────────────────┘  │                   │       ║
║  └──────────────────────────────────────────────────────┘  │                   │       ║
║                                                             │                   │       ║
║  ┌──────────────────────────────────────────────────────┐  │                   │       ║
║  │            Network Manager                            │  │                   │       ║
║  │         (utils/network_manager.py)                    │  │                   │       ║
║  │   OSMnx road-graph  ·  LRU cache (cap=3)             │  │                   │       ║
║  │   Daemon prefetch thread on AOI confirm               │  │                   │       ║
║  └──────────────────────────────────────────────────────┘  │                   │       ║
║                                                             │                   │       ║
║  ┌──────────────────────────────────────────────────────┐  │                   │       ║
║  │      Post-Solve Analytics & Reproducibility           │  │                   │       ║
║  │                                                       │  │                   │       ║
║  │  ┌──────────────────┐   ┌──────────────────────────┐ │  │                   │       ║
║  │  │  equity_metrics  │   │  repro_logger            │ │  │                   │       ║
║  │  │  (Gini, top-     │   │  (JSON record per run    │ │  │                   │       ║
║  │  │   decile share)  │   │   to RUNS_DIR)           │ │  │                   │       ║
║  │  └──────────────────┘   └──────────────────────────┘ │  │                   │       ║
║  └──────────────────────────────────────────────────────┘  │                   │       ║
╚════════════════════════════════════════════════════════════╪═══════════════════╪═══════╝
                                                             │                   │
╔════════════════════════════════════════════════════════════╪═══════════════════╪═══════╗
║                    EXTERNAL DATA LAYER                      │                   │       ║
║                                                             │                   │       ║
║  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  ┌─▼─────────────────┐ │       ║
║  │ Google Gemini│  │ Overture Maps │  │    HDX       │  │   OpenStreetMap   │ │       ║
║  │  (via ADK)   │  │ (S3 DuckDB   │  │ Population   │  │   OSMnx road      │ │       ║
║  │              │  │  queries)     │  │  grids       │  │   graph API       │ │       ║
║  └──────────────┘  └───────────────┘  └──────────────┘  └───────────────────┘ │       ║
║                                                                                 │       ║
║  ┌──────────────┐  ┌───────────────┐                                           │       ║
║  │  Nominatim   │  │   Overpass    │                                           │       ║
║  │  (geocoding  │  │  (OSM POIs    │                                           │       ║
║  │   fallback)  │  │   fallback)   │                                           │       ║
║  └──────────────┘  └───────────────┘                                           │       ║
╚═════════════════════════════════════════════════════════════════════════════════╪═══════╝
```

---

## Figure 2 — Request / Response Sequence (Chat Turn)

```
  Client (Next.js)          FastAPI            SOCAAgent          Gemini (ADK)
       │                       │                    │                    │
       │  POST /api/chat/stream│                    │                    │
       │  X-Session-Id: <id>  │                    │                    │
       │──────────────────────►│                    │                    │
       │                       │ resolve session    │                    │
       │                       │ (SessionStore)     │                    │
       │                       │──────────────────► │                    │
       │                       │ set_current_context│                    │
       │                       │ (StateBridge)      │                    │
       │                       │                    │ Runner.run()       │
       │                       │                    │───────────────────►│
       │                       │                    │                    │ function call
       │  text/event-stream    │                    │◄── tool_call ──────│
       │◄──────────────────────│◄─── SSE frame ─────│                    │
       │  (token / tool event) │                    │ execute tool()     │
       │                       │                    │ update problem_    │
       │                       │                    │ state              │
       │                       │                    │ emit → EventBus    │
       │◄──────────────────────│◄─── SSE frame ─────│                    │
       │  (activity-log event) │  (event_bus fan)   │                    │
       │                       │                    │ tool result ──────►│
       │                       │                    │                    │ next token
       │  SSE: token chunks    │                    │◄─── text_chunk ────│
       │◄──────────────────────│◄─── SSE frame ─────│                    │
       │                       │                    │                    │
       │  SSE: [DONE]          │                    │                    │
       │◄──────────────────────│◄─── stream close ──│                    │
       │                       │                    │                    │
```

---

## Figure 3 — Component Dependency Graph

```
                     ┌─────────────────────────────────────┐
                     │           backend/main.py            │
                     │  FastAPI app + CORS + lifespan       │
                     └──┬──────────────────────────────┬───┘
                        │                              │
           ┌────────────▼────────────┐    ┌────────────▼────────────┐
           │   backend/api/*.py      │    │  backend/services/*.py  │
           │   (9 routers)           │    │  SessionStore           │
           │                         │    │  EventBus               │
           └────────────┬────────────┘    └────────────┬────────────┘
                        │                              │
                        └──────────────┬───────────────┘
                                       │
                           ┌───────────▼──────────────┐
                           │     agent/soca_agent.py   │
                           │     SOCAAgent             │
                           │     ADK LlmAgent + Runner │
                           └───────────┬───────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                   │
         ┌──────────▼──────┐  ┌───────▼────────┐  ┌──────▼───────────┐
         │  fetch_tools.py │  │ optimize_tools │  │ sensitivity_     │
         │  fetch_city_    │  │ stage_ /       │  │ tools.py         │
         │  data           │  │ confirm_       │  │ run_sensitivity_ │
         └──────────┬──────┘  └───────┬────────┘  │ analysis         │
                    │                  │           └──────────────────┘
         ┌──────────▼──────┐  ┌───────▼────────────────────────────┐
         │ utils/fetchers/ │  │         solvers/registry.py        │
         │ DataFetcher     │  │                                     │
         │ (facade.py)     │  │  P-Median / P-Center / MCLP / LSCP │
         └──────────┬──────┘  └───────┬────────────────────────────┘
                    │                  │
         ┌──────────▼──────┐  ┌───────▼────────┐
         │ boundaries.py   │  │ Gurobi / PuLP  │
         │ pois.py         │  │ (MIP solver)   │
         │ population.py   │  └────────────────┘
         │ overture_duckdb │
         │ http.py         │
         └─────────────────┘
```

---

## Figure 4 — Session State Lifecycle

```
  POST /api/session
        │
        ▼
  SessionStore.create()
  ┌─────────────────────────────────────────────────────────────────┐
  │  problem_state = {                                               │
  │    problem_type:          None                                   │
  │    parameters:            {}                                     │
  │    constraints:           {}                                     │
  │    data:                  {}   ← populated by fetch_city_data   │
  │    solution:              None ← populated by confirm_optim.    │
  │    solution_history:      []                                     │
  │    pending_action:        None ← set by stage_optimization      │
  │    parameters_confirmed:  False                                  │
  │  }                                                               │
  └─────────────────────────────────────────────────────────────────┘
        │
        ├─ fetch_city_data ────────── data["boundary_*"]  ← boundaries
        │                             data["demand_*"]    ← population
        │                             data["*_pois_*"]   ← POIs
        │
        ├─ stage_optimization ─────── pending_action = { type, params }
        │                             parameters_confirmed = False
        │
        ├─ confirm_optimization ───── pending_action cleared
        │                             solution = { status, objective,
        │                                          selected_facilities,
        │                                          assignments, metrics,
        │                                          equity_metrics }
        │                             solution_history.append(solution)
        │                             repro_logger writes JSON to RUNS_DIR
        │
        └─ run_sensitivity_analysis ─ solution["sensitivity"] appended
```

---

## Table 1 — API Endpoint Reference

| Method | Endpoint               | Transport   | Purpose                                      |
|--------|------------------------|-------------|----------------------------------------------|
| POST   | `/api/session`         | REST        | Allocate session; returns `X-Session-Id`     |
| POST   | `/api/chat/stream`     | SSE         | Agent turn; streams tokens + tool events     |
| GET    | `/api/events/stream`   | SSE         | Activity-log event fan-out                   |
| POST   | `/api/aoi`             | REST        | Set area-of-interest geometry                |
| GET    | `/api/network`         | REST        | Road-network prefetch status                 |
| GET    | `/api/map`             | REST        | MapLibre GL layer payloads                   |
| POST   | `/api/data`            | REST        | Upload GeoJSON / Shapefile / CSV             |
| GET    | `/api/problems`        | REST        | List registered solver types                 |
| GET    | `/api/export`          | REST        | Download solution as GeoJSON / CSV           |
| GET    | `/api/health`          | REST        | Liveness probe                               |

---

## Table 2 — ADK Tool Reference

| Tool Function              | Trigger Condition                        | Primary Effect                                          |
|----------------------------|------------------------------------------|---------------------------------------------------------|
| `fetch_city_data`          | User names a location                    | Populates `data` with boundaries, population, POIs      |
| `get_data_status`          | User asks about loaded data              | Returns summary of `problem_state["data"]`              |
| `stage_optimization`       | Parameters confirmed, data loaded        | Writes `pending_action`; prompts user for confirmation  |
| `confirm_optimization`     | User explicitly approves                 | Invokes solver; attaches equity metrics; logs to disk   |
| `run_sensitivity_analysis` | Post-solve user request                  | Drop-one re-solves; reports per-facility degradation    |

---

*Generated 2026-05-12. Reflects commit `640375d`.*
