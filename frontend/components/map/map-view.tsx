"use client";

import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import type { MapLayer, MapState } from "@/types";
import { cn } from "@/lib/cn";
import { formatNumber } from "@/lib/format";

const BASEMAP =
  "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

// Layer render order (bottom → top)
const ROLE_ORDER: MapLayer["role"][] = [
  "assignment",
  "demand",
  "candidate",
  "boundary",
  "other",
  "selected",
];

function sorted(layers: MapLayer[]): MapLayer[] {
  return [...layers].sort(
    (a, b) => ROLE_ORDER.indexOf(a.role) - ROLE_ORDER.indexOf(b.role),
  );
}

function layerSpecs(
  layer: MapLayer,
): maplibregl.LayerSpecification[] {
  const { id, role } = layer;
  switch (role) {
    case "boundary":
      return [
        {
          id: `${id}--fill`,
          type: "fill",
          source: id,
          paint: { "fill-color": "#1565C0", "fill-opacity": 0.05 },
        } as maplibregl.FillLayerSpecification,
        {
          id: `${id}--line`,
          type: "line",
          source: id,
          paint: {
            "line-color": "#1565C0",
            "line-width": 1.5,
            "line-opacity": 0.75,
          },
        } as maplibregl.LineLayerSpecification,
      ];
    case "demand":
      return [
        {
          id: `${id}--circle`,
          type: "circle",
          source: id,
          paint: {
            "circle-color": "#4285F4",
            "circle-radius": 3,
            "circle-opacity": 0.65,
            "circle-stroke-width": 0,
          },
        } as maplibregl.CircleLayerSpecification,
      ];
    case "candidate":
      return [
        {
          id: `${id}--circle`,
          type: "circle",
          source: id,
          paint: {
            "circle-color": "#9E9E9E",
            "circle-radius": 4,
            "circle-opacity": 0.5,
            "circle-stroke-width": 1,
            "circle-stroke-color": "#757575",
            "circle-stroke-opacity": 0.6,
          },
        } as maplibregl.CircleLayerSpecification,
      ];
    case "selected":
      return [
        {
          id: `${id}--circle`,
          type: "circle",
          source: id,
          paint: {
            "circle-color": "#B5482E",
            "circle-radius": 8,
            "circle-opacity": 1,
            "circle-stroke-width": 2,
            "circle-stroke-color": "#ffffff",
          },
        } as maplibregl.CircleLayerSpecification,
      ];
    case "assignment":
      return [
        {
          id: `${id}--line`,
          type: "line",
          source: id,
          paint: {
            "line-color": "#9E9E9E",
            "line-width": 0.75,
            "line-opacity": 0.45,
          },
        } as maplibregl.LineLayerSpecification,
      ];
    default:
      return [
        {
          id: `${id}--circle`,
          type: "circle",
          source: id,
          paint: {
            "circle-color": "#607D8B",
            "circle-radius": 3,
            "circle-opacity": 0.6,
          },
        } as maplibregl.CircleLayerSpecification,
      ];
  }
}

/**
 * Remove all managed sources (those whose IDs appear in `managedSources`)
 * and their dependent layers from the map, then clear the set.
 */
function clearManagedLayers(
  map: maplibregl.Map,
  managedSources: Set<string>,
): void {
  const style = map.getStyle();
  for (const sourceId of [...managedSources]) {
    for (const layer of style?.layers ?? []) {
      if ((layer as { source?: string }).source === sourceId) {
        try {
          map.removeLayer(layer.id);
        } catch {}
      }
    }
    try {
      map.removeSource(sourceId);
    } catch {}
  }
  managedSources.clear();
}

function applyLayers(
  map: maplibregl.Map,
  layers: MapLayer[],
  managedSources: Set<string>,
): void {
  clearManagedLayers(map, managedSources);
  for (const layer of sorted(layers)) {
    try {
      map.addSource(layer.id, {
        type: "geojson",
        data: layer.geojson,
      });
      managedSources.add(layer.id);
      for (const spec of layerSpecs(layer)) {
        map.addLayer(spec);
      }
    } catch (e) {
      console.warn("[MapView] failed to add layer", layer.id, e);
    }
  }
}

interface Props {
  state: MapState;
}

export function MapView({ state }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const managedSources = useRef<Set<string>>(new Set());
  const loadedRef = useRef(false);
  // Track last view state to avoid unnecessary flyTo calls
  const lastView = useRef({ lon: NaN, lat: NaN });

  // Initialise map once on mount
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const { longitude, latitude, zoom } = state.view_state;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASEMAP,
      center: [longitude, latitude],
      zoom,
      attributionControl: false,
      dragRotate: false,
      touchPitch: false,
    });

    map.addControl(
      new maplibregl.AttributionControl({ compact: true }),
      "bottom-right",
    );
    map.addControl(
      new maplibregl.NavigationControl({ showCompass: false }),
      "top-right",
    );

    mapRef.current = map;
    lastView.current = { lon: longitude, lat: latitude };

    map.on("load", () => {
      loadedRef.current = true;
      applyLayers(map, state.layers, managedSources.current);
    });

    return () => {
      loadedRef.current = false;
      map.remove();
      mapRef.current = null;
      managedSources.current.clear();
    };
    // Run only on mount — subsequent state changes are handled below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync layers whenever state.layers changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;
    applyLayers(map, state.layers, managedSources.current);
  }, [state.layers]);

  // Fly to new location when view_state changes meaningfully
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const { longitude, latitude, zoom } = state.view_state;
    const prev = lastView.current;
    const moved =
      Math.abs(longitude - prev.lon) > 0.001 ||
      Math.abs(latitude - prev.lat) > 0.001;
    if (!moved) return;
    lastView.current = { lon: longitude, lat: latitude };
    map.flyTo({ center: [longitude, latitude], zoom, duration: 800 });
  }, [state.view_state]);

  const sol = state.solution;

  return (
    <div className="relative h-full w-full overflow-hidden">
      <div ref={containerRef} className="h-full w-full" />

      {/* Legend */}
      <div className="absolute bottom-8 left-3 z-10 flex flex-col gap-1 rounded border border-border bg-surface/90 px-2 py-1.5 text-2xs text-text-muted shadow-popover backdrop-blur">
        {state.layers.some((l) => l.role === "demand") && (
          <LegendRow color="#4285F4" label="Demand" />
        )}
        {state.layers.some((l) => l.role === "candidate") && (
          <LegendRow color="#9E9E9E" label="Candidates" />
        )}
        {state.layers.some((l) => l.role === "selected") && (
          <LegendRow color="#B5482E" label="Selected facilities" />
        )}
        {state.layers.some((l) => l.role === "assignment") && (
          <LegendRow color="#9E9E9E" label="Assignments" line />
        )}
      </div>

      {/* Solution metrics overlay */}
      {sol && (
        <div className="absolute left-3 top-3 z-10 max-w-[220px] rounded border border-border bg-surface/90 px-2.5 py-2 text-xs text-text shadow-popover backdrop-blur">
          <div className="mono text-2xs font-medium uppercase tracking-wider text-text-faint">
            {sol.problem_type ?? "solution"}
          </div>
          <div className="mt-1 space-y-0.5">
            <MetricRow
              label="Facilities"
              value={String(sol.n_selected)}
            />
            {sol.objective_value != null && (
              <MetricRow
                label="Objective"
                value={formatNumber(sol.objective_value)}
              />
            )}
            {Object.entries(sol.metrics)
              .slice(0, 3)
              .map(([k, v]) => (
                <MetricRow
                  key={k}
                  label={k.replace(/_/g, " ")}
                  value={
                    typeof v === "number"
                      ? formatNumber(v as number)
                      : String(v)
                  }
                />
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

function LegendRow({
  color,
  label,
  line,
}: {
  color: string;
  label: string;
  line?: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      {line ? (
        <div className="h-px w-4 shrink-0" style={{ background: color }} />
      ) : (
        <div
          className="h-2.5 w-2.5 shrink-0 rounded-full"
          style={{ background: color }}
        />
      )}
      <span>{label}</span>
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className={cn("flex items-baseline justify-between gap-3")}>
      <span className="text-text-muted">{label}</span>
      <span className="mono text-text">{value}</span>
    </div>
  );
}
