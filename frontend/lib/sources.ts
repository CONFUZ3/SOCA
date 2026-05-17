/**
 * Canonical source-name → human label mapping for data provenance.
 * Keys match the `source` strings emitted by backend fetchers
 * (`utils/fetchers/*`) and `utils/activity_log.py`.
 */

const LABELS: Record<string, string> = {
  overture: "Overture Maps",
  "overture-maps": "Overture Maps",
  nominatim: "OpenStreetMap (Nominatim)",
  photon: "Photon",
  overpass: "OpenStreetMap (Overpass)",
  osmnx: "OSMnx / OpenStreetMap",
  kontur: "Kontur Population",
  hdx: "HDX (Humanitarian Data Exchange)",
  "kontur-hdx": "Kontur · HDX",
  hdx_facebook_population: "HDX population",
  hdx_kontur_population: "Kontur · HDX population",
  auto_fetched: "Auto-fetched",
  user: "Uploaded by user",
  local: "Local file",
  synthetic: "Synthetic (fallback)",
  synthetic_uniform_grid: "Synthetic population grid",
  generated: "Generated",
};

export function sourceLabel(source: string | null | undefined): string {
  if (!source) return "Unknown source";
  const key = source.toLowerCase();
  return LABELS[key] || source;
}
