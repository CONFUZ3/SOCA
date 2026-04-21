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
  geometry_type: string;
  columns: string[];
  bounds: number[];
  source?: string | null;
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

export interface MapSolution {
  status: string;
  objective_value: number | null;
  metrics: Record<string, unknown>;
  n_selected: number;
  problem_type: string | null;
}

export interface MapState {
  view_state: MapViewState;
  layers: MapLayer[];
  solution: MapSolution | null;
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
