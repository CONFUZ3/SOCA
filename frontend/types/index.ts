/**
 * Wire types mirroring the Pydantic / dict shapes returned by the FastAPI
 * backend. Kept in one file for now — split as it grows.
 */

export type ProblemType = "p-median" | "p-center" | "mclp" | "lscp";

export interface AoiInfo {
  name: string;
  source: string;
  area_km2: number;
  geometry: unknown; // GeoJSON FeatureCollection / Feature / Geometry
}

export interface DatasetSummary {
  name: string;
  num_features: number;
  active_num_features?: number;
  geometry_type: string;
  columns: string[];
  bounds: number[];
  source?: string | null;
  role?: "boundary" | "demand" | "candidate" | "other";
  source_details?: string[];
  numeric_preview?: Record<string, number>;
  numeric_summary?: Array<{
    key: string;
    label: string;
    value: number;
    stat: "total" | "mean" | string;
  }>;
  available_subcategories?: string[];
  active_subcategories?: string[];
  subcategory_counts?: Record<string, number>;
}

export interface NetworkState {
  status: null | "fetching" | "ready" | "failed";
  error?: string | null;
  stats?: { nodes?: number; edges?: number } | null;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  tool_calls?: string[];
}

export interface SessionSnapshot {
  aoi: AoiInfo | null;
  aoi_confirmed: boolean;
  problem_type: ProblemType | null;
  parameters: Record<string, unknown>;
  constraints: Record<string, unknown>;
  datasets: DatasetSummary[];
  has_solution: boolean;
  solution_status?: string | null;
  solution_version?: number;
  messages: ChatMessage[];
  network: NetworkState;
  settings: {
    generated_sites_count: number;
    generated_sites_seed: number | null;
  };
}

export interface GeocodeCandidate {
  display_name: string;
  short_name: string;
  context: string;
  kind: string;
  lat: number;
  lon: number;
  bbox?: [number, number, number, number] | null;
  osm_type?: string | null;
  osm_id?: number | null;
  place_rank: number;
  country: string;
  source: string;
  has_relation?: boolean;
}

export interface ActivityEvent {
  stage: string;
  status: "try" | "info" | "ok" | "fail";
  detail: string;
  source?: string | null;
  duration_ms?: number | null;
  timestamp: number;
  extra?: Record<string, unknown>;
}

export interface NetworkSseEvent {
  status: "fetching" | "ready" | "failed" | "idle" | string;
  error?: string | null;
  stats?: { nodes?: number; edges?: number };
}

export interface ToolCallStart {
  name: string;
  args: Record<string, unknown>;
}

export interface ToolCallResult {
  name: string;
  summary: Record<string, unknown>;
}

export type MapLayerRole =
  | "boundary"
  | "demand"
  | "candidate"
  | "selected"
  | "assignment"
  | "coverage"
  | "access_heatmap"
  | "facility_coverage"
  | "facility_gaps"
  | "other";

export interface MapViewState {
  longitude: number;
  latitude: number;
  zoom: number;
}

export interface MapLayer {
  id: string;
  role: MapLayerRole;
  geojson: GeoJSON.FeatureCollection;
}

/**
 * Mirrors the curated solution payload emitted by
 * ``backend/api/map.py::_build_solution_summary``.
 *
 * ``solver`` / ``solver_time_seconds`` / ``distance_metric_used`` /
 * ``service_radius_m`` / ``warnings`` come from the run-context injected
 * by ``confirm_optimization`` before the solution is persisted in
 * ``problem_state["solution"]``.
 */
export interface MapSolution {
  status: string;
  problem_type: string | null;
  variant?: string | null;
  objective_value: number | null;
  metrics: Record<string, unknown>;
  n_selected: number;
  solver?: string | null;
  solver_time_seconds?: number | null;
  gap?: number | null;
  distance_metric_used?: "network" | "euclidean" | string | null;
  service_radius_m?: number | null;
  warnings?: string[];
}

export interface AnalysisWorstPoint {
  demand_idx: number;
  distance_m: number;
  weight: number;
  lat: number;
  lon: number;
}

export interface MapAnalysis {
  facility_dataset_key: string | null;
  service_radius_m: number;
  coverage?: {
    pct_demand_covered: number;
    pct_points_covered: number;
    uncovered_demand_weight: number;
    service_radius_m: number;
  };
  access?: {
    avg_distance_m: number;
    max_distance_m: number;
    p90_distance_m: number;
    gini_coefficient: number;
    bottom_decile_avg_distance_m: number;
  };
  density?: {
    facilities_per_km2: number;
    facilities_per_1000_people: number;
    area_km2: number;
    n_facilities: number;
    n_demand_points: number;
  };
  distance_metric_used?: string;
  warnings?: string[];
  spatial_breakdown?: {
    worst_access_points: AnalysisWorstPoint[];
    n_uncovered_demand_points: number;
  };
}

export interface MapState {
  view_state: MapViewState;
  layers: MapLayer[];
  solution: MapSolution | null;
  analysis?: MapAnalysis | null;
}

export interface ProblemInfo {
  short_name: ProblemType;
  name: string;
  category: string;
  description: string;
  keywords: string[];
  variants: string[];
  complexity: string;
  typical_use_cases: string[];
  conversation_prompts: Record<string, unknown>;
  visualization_config: Record<string, unknown>;
}
