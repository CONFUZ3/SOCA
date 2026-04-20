# SOCA — Streamlit → React Migration Plan

> A staged plan to replace the Streamlit UI in `app.py` with a modern React
> front-end while preserving every solver, agent, and data-fetching capability
> that already exists in the Python codebase.

---

## 1. Goals & Non-Goals

### Goals

- Replace `app.py` (Streamlit) with a production-grade React SPA.
- Keep **all** existing Python logic (`agent/`, `solvers/`, `utils/`) reusable behind
  a thin FastAPI service. No re-implementation of optimisation or data-fetching
  logic in JS.
- **Real-time agent transparency**: stream tool calls (`fetch_city_data`,
  `stage_optimization`, `confirm_optimization`, `get_data_status`) and
  activity-log events (`utils/activity_log.py`) to the browser as they happen,
  not after the turn completes.
- **Great UI/UX**: a split chat + map layout that is fast (WebGL map,
  virtualised chat), accessible (WCAG AA, full keyboard), and responsive
  (desktop-first, collapses gracefully on tablets).
- **Feature parity** on day one: AOI selection, file upload, raster upload,
  problem configuration, solution visualisation, metrics panel, exports
  (GeoJSON/CSV/PDF), road-network status, solver settings.

### Non-Goals (intentional omissions)

- No multi-user backend (auth, RBAC, persistence). Current Streamlit app is
  effectively single-session; keep that contract in v1. Add a pluggable
  `SessionStore` interface so multi-user + Postgres can be added later
  without UI changes.
- No mobile-first redesign. A 13"+ screen is the primary target; mobile is a
  "does not break" requirement only.
- No rewrite of the optimisation or data-fetching layers.

---

## 2. Current State — What We Are Replacing

### 2.1 Streamlit surface area in `app.py`

| Region | Widget / Fragment | React replacement |
| --- | --- | --- |
| AOI gate (pre-flow) | `render_aoi_selector()` (folium + Draw) | React Leaflet + `leaflet-draw` or MapLibre + MapboxDraw, with Photon autocomplete served by backend. |
| AOI header chip | `st.markdown` + button | `<AoiHeader />` component. |
| Road-network banner | `st.info` / `st.warning` / `st.caption` | `<NetworkStatusBanner />` driven by SSE events. |
| Sidebar — file uploaders | `st.file_uploader` (vector + raster) | `react-dropzone` with drag-and-drop, progress, per-file status. |
| Sidebar — renderer/basemap selectors | `st.radio` / `st.selectbox` | Headless UI `RadioGroup` / `Listbox`. |
| Sidebar — loaded data/raster expanders | `st.expander` | Accordion components with dataset cards. |
| Sidebar — candidate generator controls | Number input + seed | Numeric input + seed popover. |
| Chat column | `st.chat_message`, `st.status`, `st.chat_input` | `<ChatPanel />` with virtualised message list, streaming token renderer, inline tool-call bubbles, `<AgentStatusDrawer />` for live activity log. |
| Map column | `render_map_fragment()` (PyDeck in iframe) | `<MapCanvas />` using `deck.gl` (`@deck.gl/react`) + MapLibre basemap. Native React — no HTML iframe. |
| Metrics dashboard | Per-problem `_display_*_metrics` | `<MetricsPanel />` with problem-type dispatch. |
| Export buttons | `st.download_button` | `<ExportMenu />` calling backend export endpoints, using `fetch` + object URLs. |
| Reset / Change AOI | `st.button` + `st.rerun` | Modal confirm → dispatch reset action. |

### 2.2 Python entry points to preserve

- `SOCAAgent.chat()` / `SOCAAgent.notify_data_uploaded()` → wrap with an async
  streaming variant (`chat_stream()`).
- `utils/data_fetcher.DataFetcher` → unchanged; called from ADK tools.
- `utils/aoi_selector` → the _folium rendering_ is Streamlit-specific, but
  helpers (`aoi_to_boundary_gdf`, `_area_km2`, `_simplify_for_edit`) are
  reusable. Extract pure geometry helpers from Streamlit-dependent rendering.
- `utils/geocoder.suggest()` → exposed as `GET /api/geocode/suggest`.
- `utils/network_manager.NetworkManager` + `launch_prefetch_thread` → drive
  the "fetching/ready/failed" state over SSE.
- `utils/activity_log` → augment `log_event()` to also push into an
  `asyncio.Queue` registered per session, so the API layer can stream events.
- `solvers/registry.problem_registry` → exposed verbatim via
  `GET /api/problems`.
- `utils/export_handler.ExportHandler` → three download endpoints.

---

## 3. Target Architecture

```
┌──────────────────────── Browser ────────────────────────┐
│  React SPA (Vite + TS)                                  │
│   - TanStack Query for REST                             │
│   - EventSource for SSE streams                         │
│   - Zustand for client state                            │
│   - deck.gl + MapLibre for map                          │
└────────┬───────────────────────────────┬────────────────┘
         │ REST (JSON)                   │ SSE (text/event-stream)
         ▼                               ▼
┌────────────────────────── FastAPI ──────────────────────┐
│  /api/session            POST, GET                      │
│  /api/aoi/suggest        GET   → utils.geocoder         │
│  /api/aoi/boundary       POST  → osm relation → GeoJSON │
│  /api/aoi/confirm        POST  → sets AOI, kicks graph  │
│  /api/data/upload        POST multipart → DataProcessor │
│  /api/chat/stream        POST → SSE (tool/token events) │
│  /api/solve/status       GET   → last run details       │
│  /api/export/{fmt}       GET   → GeoJSON/CSV/PDF bytes  │
│  /api/events/stream      GET   → SSE activity_log       │
│  /api/network/status     GET   → prefetch state         │
│  /api/network/refresh    POST                           │
└────────┬────────────────────────────────────────────────┘
         │ direct imports (same process)
         ▼
┌─────────────────────────────────────────────────────────┐
│  SOCAAgent · problem_registry · DataFetcher · solvers   │
│  ← UNCHANGED; the UI boundary is the only thing moving. │
└─────────────────────────────────────────────────────────┘
```

### 3.1 Why FastAPI (and not keep Streamlit + React on top)

- FastAPI already works with Pydantic / async generators / SSE.
- ADK's `Runner.run_async()` is an **async generator** — a natural fit for
  SSE streaming. No need for polling or adapters.
- Same Python process → we can import `SOCAAgent`, `problem_registry`,
  `DataFetcher` directly. No IPC. No duplicated data models.
- Replaces Streamlit session state with an explicit `SessionStore`, which
  makes the state model testable and observable.

### 3.2 Session model

- Each browser gets a `session_id` (cookie, HttpOnly, SameSite=Lax) on first
  request.
- Server-side `SessionStore` (in-memory dict keyed by `session_id`)
  holds the exact same keys as today's `st.session_state.problem_state`
  plus: `messages`, `raster_data`, `_activity_log`, `_network_status`, etc.
- The `state_bridge` in `agent/tools/state_bridge.py` is already
  thread-local — we only need to swap the _source_ of the bridge from
  `st.session_state` to `SessionStore.get(session_id)`. This is a
  low-risk, additive change.

### 3.3 Real-time streaming — two separate SSE channels

1. **Chat stream** (`POST /api/chat/stream` → SSE response):
   - `event: token`  — incremental assistant text from the model.
   - `event: tool_call_start` — payload: tool name + args.
   - `event: tool_call_result` — payload: tool name + summary/status.
   - `event: state_patch` — JSON Patch (RFC 6902) of diffs to
     `problem_state` (e.g. new dataset keys, AOI confirmed).
   - `event: final` — terminating marker + final text + tool_calls list.
   - `event: error`.

2. **Ambient activity stream** (`GET /api/events/stream`):
   - One long-lived SSE per session, multiplexing:
     - `utils.activity_log` events (`geocode.suggest`, `boundary.fetch`,
       `population.fetch`, `network.fetch`, …) with glyph, status, duration.
     - Network-prefetch transitions (`fetching` → `ready` / `failed`).
   - Drives an always-visible "Agent Activity" drawer so the user sees
     exactly what sources served them, even when they aren't chatting
     (e.g. when the AOI confirmation triggers a background road-graph
     fetch).

### 3.4 Backpressure & cancellation

- FastAPI SSE endpoints use `asyncio.CancelledError` on client disconnect to
  stop the Runner cleanly (wire into `runner.run_async`'s `asyncio.Task`).
- Chat stream is single-flight per session — a new `POST /chat/stream`
  cancels the previous one server-side.

---

## 4. Front-end Stack Decisions

| Concern | Choice | Why |
| --- | --- | --- |
| Bundler | **Vite** | Fast HMR, first-class TS, minimal config. |
| Language | **TypeScript (strict)** | Agent/solver contracts are best expressed as TS types. |
| UI primitives | **shadcn/ui** (Radix + Tailwind) | Accessible, copy-own, easy to theme. |
| Styling | **Tailwind CSS** + CSS variables | Fast iteration + dark-mode-ready. |
| State — server | **TanStack Query** | Deduped fetches, caching, retries. |
| State — client | **Zustand** | Small, no context boilerplate, great for a multi-panel app. |
| Forms | **react-hook-form** + **zod** | Validation mirrors solver parameter schemas. |
| Map | **deck.gl** (`@deck.gl/react`) + **MapLibre GL** | Same engine as the current PyDeck, WebGL perf, GeoJSON layers natively. |
| Draw tool | **maplibre-gl-draw** (or **react-leaflet-draw** as fallback) | Parity with `folium.plugins.Draw`. |
| Chat rendering | **react-markdown** + **remark-gfm** + **rehype-highlight** | Matches current `st.markdown` rendering. |
| Virtualisation | **@tanstack/react-virtual** | Keep chat fast after 500+ messages. |
| Icons | **lucide-react** | Matches shadcn idioms. |
| Toasts | **sonner** | Replaces `st.toast`. |
| Dates/format | **date-fns**, **Intl.NumberFormat** | Metric formatting. |
| PDF export | Backend-generated (existing ReportLab). Browser just downloads the blob. | No JS duplication. |
| Testing | **Vitest** + **React Testing Library**; **Playwright** for e2e | Same philosophy as existing pytest. |

### Folder layout (new)

```
frontend/
  ├─ src/
  │  ├─ app/                # routes / top-level shells
  │  ├─ components/
  │  │  ├─ aoi/             # AoiSelector, AoiHeader
  │  │  ├─ chat/            # ChatPanel, MessageBubble, ToolCallBubble, AgentStatusDrawer
  │  │  ├─ map/             # MapCanvas, LayerSelector, LegendOverlay
  │  │  ├─ metrics/         # MetricsPanel + per-problem renderers
  │  │  ├─ data/            # UploadDropzone, DatasetList, RasterList, CandidateGenOptions
  │  │  ├─ export/          # ExportMenu
  │  │  └─ ui/              # shadcn primitives
  │  ├─ hooks/              # useChatStream, useActivityStream, useNetworkStatus, …
  │  ├─ lib/
  │  │  ├─ api.ts           # fetch wrappers
  │  │  ├─ sse.ts           # typed EventSource helper
  │  │  ├─ problem-state.ts # zustand store + JSON-Patch reducer
  │  │  └─ geo.ts           # area_km2 etc. (or delegate to backend)
  │  ├─ types/              # ProblemType, Solution, Dataset, ChatEvent…
  │  └─ styles/
  ├─ tests/                 # vitest + playwright
  ├─ public/
  ├─ index.html
  ├─ package.json
  ├─ tsconfig.json
  ├─ vite.config.ts
  └─ tailwind.config.ts
```

---

## 5. Backend (FastAPI) — Detailed Endpoints

> All endpoints accept/issue `session_id` via an HttpOnly cookie. Example
> shapes below are compact for readability.

### 5.1 Session

- `POST /api/session` → `{ session_id, created_at }`. Idempotent.
- `GET /api/session` → current `ProblemStateDTO`.

```ts
type ProblemStateDTO = {
  aoi: { name: string; area_km2: number; source: string; geometry: GeoJSON } | null;
  aoi_confirmed: boolean;
  problem_type: "p-median" | "p-center" | "mclp" | "lscp" | null;
  parameters: Record<string, unknown>;
  constraints: Record<string, unknown>;
  datasets: DatasetSummary[];          // role-classified (demand|candidate|boundary|poi)
  solution: SolutionDTO | null;
  messages: ChatMessage[];
  network_status: "idle" | "fetching" | "ready" | "failed";
  network_stats?: { nodes: number; edges: number };
};
```

### 5.2 AOI

- `GET /api/aoi/suggest?q=Brook&limit=6` → `[GeocodeCandidate]`.
- `POST /api/aoi/boundary` `{ osm_id } | { geojson }` → resolved boundary GeoJSON + `area_km2` + source label.
- `POST /api/aoi/confirm` `{ name, geojson, source }` → sets AOI,
  launches `launch_prefetch_thread` in background, returns DTO.

### 5.3 Data

- `POST /api/data/upload` (multipart) — accepts the same file types as
  `st.file_uploader`. Re-uses `DataProcessor.load_file` +
  `preprocess_data`. Returns classified summaries. Triggers
  `SOCAAgent.notify_data_uploaded` via the chat stream.
- `POST /api/data/raster/upload` — parity with existing raster uploader.
- `DELETE /api/data/{dataset_name}` — remove dataset.
- `GET /api/data/{dataset_name}.geojson` — map layer payload.

### 5.4 Chat

- `POST /api/chat/stream` `{ message }` → `text/event-stream`. See §3.3.
- `GET /api/chat/history` → recent `ChatMessage[]` (from `SessionStore`).

Server-side implementation:

```python
async def chat_stream(session_id: str, msg: str):
    ps = SessionStore.get(session_id)
    set_current_context(data=ps["data"], problem_state=ps, ...)
    runner = get_runner()
    async for event in runner.run_async(
        user_id=session_id,
        session_id=ps["_adk_session_id"],
        new_message=build_content(msg, ps),
    ):
        for sse in translate_adk_event(event):  # token/tool_call/etc
            yield sse
    yield sse_final(ps)
```

### 5.5 Events (ambient)

- `GET /api/events/stream` → SSE. Server subscribes to
  `activity_log` and `network_status` queues for the session and forwards
  each event as JSON.

Requires one additive change to `utils/activity_log.py`: optionally
publish events to an `asyncio.Queue` registered via
`activity_log.register_sink(session_id, queue)`. Zero changes to call sites.

### 5.6 Exports

- `GET /api/export/geojson` → bytes.
- `GET /api/export/csv` → bytes.
- `GET /api/export/pdf` → bytes.

Re-uses `ExportHandler` verbatim.

### 5.7 Network

- `GET /api/network/status` → `{ status, stats, error }`.
- `POST /api/network/refresh` → re-runs `launch_prefetch_thread`.

### 5.8 Problems

- `GET /api/problems` → `problem_registry.list_problems()` + each problem's
  `get_conversation_prompts()` + `get_visualization_config()`. Consumed by
  the React UI to render parameter forms and legend metadata.

---

## 6. Real-time UX (How the user experiences the agent)

The biggest UX upgrade over Streamlit is real-time agent transparency.
Concretely:

1. User types "Place 5 hospitals in Nairobi" and hits enter.
2. Chat pane immediately shows a pending assistant bubble with a shimmering
   cursor. Below it, an inline **tool-call timeline** appears:
   - `• fetch_city_data(location="Nairobi, Kenya", include_population=True)`
     (streaming — chevron spins).
   - Sub-events from `activity_log` render as indented rows under the
     tool-call card: `… geocode.suggest Photon "Nairobi"`, then
     `✓ geocode.suggest Photon (412 ms)`, then `… boundary.fetch Overture`,
     `✓ boundary.fetch Overture (1,287 ms)`, `… population.fetch HDX`, etc.
   - As each row resolves, glyph and duration update in place.
3. Once `fetch_city_data` returns, the map layer animates in (deck.gl's
   transition API) without a full page reflow.
4. Model streams the next tokens; the user sees them appear as they arrive.
5. If `stage_optimization` then `confirm_optimization` fire, their cards
   appear next in the timeline, with a `Solving…` progress spinner tied to
   solver elapsed time.
6. When the final solution arrives, the map updates and the metrics panel
   animates from skeleton to populated.

All of this happens over two open SSE connections (`/chat/stream` and
`/events/stream`). The client keeps them healthy with auto-reconnect and
monotonic event IDs for dedupe.

### Agent Activity Drawer

A persistent right-edge drawer (collapsible) always shows the last 50
activity-log events, mirroring the existing `render_log()` expander but
live and not tied to a chat turn. Great for diagnosing AOI/road-graph
background fetches that don't belong to any chat message.

### Map

- deck.gl layers for boundary (GeoJsonLayer), demand (ScatterplotLayer or
  HeatmapLayer), candidates (ScatterplotLayer), selected facilities (IconLayer),
  assignments (LineLayer), service areas (GeoJsonLayer with buffered circles).
- MapLibre basemap with three presets (light/dark/voyager) that mirror the
  current PyDeck selector.
- Keyboard: `L` toggles layers popover, `F` fits to data, `B` toggles basemap.

---

## 7. Migration Plan — Phased Delivery

Each phase ends in a deployable app. No "big bang" cutover.

### Phase 0 — Repo prep (small, no user-visible change)

1. Introduce `backend/` package that re-exports the existing Python
   modules unchanged.
2. Add a `SessionStore` abstraction (in-memory). Leave `app.py`
   untouched.
3. Tiny additive change to `utils/activity_log.py`: optional pub/sub
   sinks.
4. CI: add Python import sanity checks + type checking on new modules.

### Phase 1 — FastAPI shell + read-only endpoints

1. `main.py` (FastAPI) with CORS, cookie middleware, session store.
2. Implement: `POST/GET /api/session`, `GET /api/problems`,
   `GET /api/aoi/suggest`, `GET /api/network/status`.
3. Unit tests per endpoint (`pytest` + `httpx.AsyncClient`).

### Phase 2 — Frontend scaffold

1. `frontend/` workspace (Vite + TS + Tailwind + shadcn).
2. API client + typed SSE helper.
3. App shell: header, split pane (chat + map placeholder), sidebar shell.
4. Wire `GET /api/problems` and `/api/network/status` to prove the
   round-trip works.
5. Dev-server proxy to FastAPI.

### Phase 3 — AOI flow

1. Backend: `POST /api/aoi/boundary`, `POST /api/aoi/confirm`.
2. Frontend: `AoiSelector` (React Leaflet + Photon autocomplete + draw),
   `AoiHeader`, `NetworkStatusBanner`.
3. Ambient SSE (`/api/events/stream`) to surface geocode + network
   events in real time during AOI selection.

### Phase 4 — Chat streaming + tool-call timeline

1. Backend: `POST /api/chat/stream` with ADK `run_async` → SSE translator.
2. Frontend: `ChatPanel`, `MessageBubble`, `ToolCallBubble`, streaming
   renderer, auto-scroll, virtualised list.
3. JSON-Patch reducer to apply `state_patch` events to the Zustand store.

### Phase 5 — Map + metrics

1. `MapCanvas` (deck.gl) with layer toggle popover and legend overlay.
2. `MetricsPanel` with per-problem renderers mirroring current Python code.
3. Smooth transitions (deck.gl transition API) when solutions update.

### Phase 6 — Data uploads + exports + raster

1. `UploadDropzone`, multipart endpoint, dataset cards.
2. Raster upload + map overlay.
3. Export menu → direct download of backend-generated bytes.
4. Candidate generator settings popover.

### Phase 7 — Polish & release

1. Accessibility sweep (Axe, keyboard, focus rings, ARIA).
2. Dark mode.
3. Error boundaries + empty/loading states.
4. E2E smoke tests (Playwright) covering the 5-step happy path.
5. Build pipeline: frontend build → FastAPI static-mount behind `/`.
6. `README`, `QUICKSTART`, and `run.sh` updates. Remove `streamlit run app.py`.

### Phase 8 — Retire Streamlit

1. Keep `app.py` in tree for one release as `legacy/app.py`.
2. Remove `streamlit` + `streamlit-folium` from required deps (move to
   `requirements-legacy.txt`).
3. Delete after one green release.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| ADK `run_async` exceptions mid-stream leaving the UI in limbo | Translate exceptions into `event: error` SSE frames; client shows red bubble + retry. |
| Large GeoDataFrames over the wire stall the map | Server-side simplification (`geopandas.simplify(tolerance)`) + Arrow/Parquet chunking if needed. |
| Streamlit session-state coupling surfaces I missed | Grep for `st.session_state` usage beyond `app.py` before starting Phase 4 — migrate call sites to `state_bridge` first. |
| Concurrent solver runs during rapid chat | Single-flight guard per session on `/chat/stream`. |
| SSE through corporate proxies/CDNs | Document reverse-proxy config; offer WebSocket fallback under `/api/ws`. |
| Multi-user support later requires a real session store | `SessionStore` is an interface from day one — swap implementation to Redis/Postgres without UI changes. |
| Accessibility regressions compared to Streamlit defaults | Use shadcn/Radix primitives; run Axe in CI; keyboard-test each phase. |
| Folium `Draw` behaviour parity (edit existing polygon) | Validate on Phase 3 demo with Brooklyn edge case; fall back to `react-leaflet-draw` if MapLibre draw misses features. |

---

## 9. Open Questions for You

Please confirm or correct the defaults below before I start Phase 0:

1. **Deployment target** — Is this staying self-hosted (Streamlit Cloud / a
   VM / Docker)? I'm planning a single-container build: FastAPI serves the
   built Vite assets. Any Kubernetes / cloud provider constraints?

2. **Multi-user** — v1 stays single-session (server-side in-memory store).
   OK, or do we need per-user persistence now?

3. **Auth** — No auth in v1 (parity with current Streamlit). Add a simple
   password / OAuth later if needed. Confirm?

4. **API surface shape** — OK with SSE for both streams, or do you want me
   to use WebSockets end-to-end? SSE is simpler and sufficient for
   one-way server-push; WS only if you anticipate bi-directional needs
   beyond chat.

5. **Styling direction** — shadcn + Tailwind is the fastest route to a
   "modern, clean" look. Any existing brand palette / logo? Dark mode a
   hard requirement or nice-to-have?

6. **Feature flag for the rollout** — Ship React at `/app` while
   Streamlit still runs at `/legacy`? Or full replacement?

7. **Drawing tool** — MapLibre + MapboxDraw is my preferred map stack
   (matches PyDeck's basemap vibe). If edit-existing-polygon parity with
   `folium.plugins.Draw` becomes painful, I'd fall back to React-Leaflet.
   OK?

8. **PDF export** — keep ReportLab on the server (my plan), or move to a
   client-rendered PDF (e.g. `pdf-lib`)? Server-side keeps one source of
   truth; client-side would allow "what you see is what you print".

Unless you tell me otherwise, I'll proceed with the defaults above and
start Phase 0 immediately.
