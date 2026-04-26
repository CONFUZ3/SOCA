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
  user: "Uploaded by user",
  local: "Local file",
  synthetic: "Synthetic (fallback)",
};

export function sourceLabel(source: string | null | undefined): string {
  if (!source) return "Unknown source";
  const key = source.toLowerCase();
  return LABELS[key] || source;
}
