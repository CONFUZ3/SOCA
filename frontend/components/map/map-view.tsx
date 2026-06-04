"use client";

import "@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css";
import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import MapboxDraw from "@mapbox/mapbox-gl-draw";
import { DRAW_STYLES } from "@/components/aoi/aoi-draw-map";
import area from "@turf/area";
import { Check, Loader2, Pencil, X } from "lucide-react";
import type { MapLayer, MapSolution, MapState } from "@/types";
import { cn } from "@/lib/cn";
import { formatNumber, formatDistanceMeters, formatArea } from "@/lib/format";
import { apiPost } from "@/lib/api";
import { useSession } from "@/hooks/use-session";
import { useStore } from "@/lib/store";
import { useQueryClient } from "@tanstack/react-query";

const BASEMAP =
  "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

// Layer render order (bottom → top).  ``coverage`` sits above the AOI but
// below demand points so population dots remain clickable.
const ROLE_ORDER: MapLayer["role"][] = [
  "coverage",
  "facility_coverage",
  "assignment",
  "demand",
  "access_heatmap",
  "facility_gaps",
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

// -----------------------------------------------------------------------
// Layer styling helpers
// -----------------------------------------------------------------------

/**
 * Compute an approximate p95 of a numeric feature property so we can
 * drive graduated symbology (demand weight, assignment stroke-width)
 * without being swamped by a handful of huge outliers.
 */
function computeP95(
  features: GeoJSON.Feature[] | undefined,
  key: string,
): number | null {
  if (!features || features.length === 0) return null;
  const vals: number[] = [];
  for (const f of features) {
    const v = (f.properties ?? {})[key];
    if (typeof v === "number" && Number.isFinite(v) && v > 0) vals.push(v);
  }
  if (vals.length === 0) return null;
  vals.sort((a, b) => a - b);
  const idx = Math.min(vals.length - 1, Math.floor(vals.length * 0.95));
  return vals[idx] ?? null;
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
    case "demand": {
      // Graduated circle radius by ``population`` / ``weight``; we read
      // a p95 threshold so a few outliers don't dominate the scale.
      const p95 = computeP95(
        (layer.geojson.features ?? []) as GeoJSON.Feature[],
        "population",
      );
      const radiusExpr: maplibregl.DataDrivenPropertyValueSpecification<number> =
        p95 && p95 > 0
          ? [
              "interpolate",
              ["linear"],
              ["coalesce", ["get", "population"], 1],
              0,
              2,
              p95,
              9,
            ]
          : 3;
      return [
        {
          id: `${id}--circle`,
          type: "circle",
          source: id,
          paint: {
            "circle-color": "#4285F4",
            "circle-radius": radiusExpr,
            "circle-opacity": 0.55,
            "circle-stroke-width": 0,
          },
        } as maplibregl.CircleLayerSpecification,
      ];
    }
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
    case "coverage":
      return [
        {
          id: `${id}--fill`,
          type: "fill",
          source: id,
          paint: {
            "fill-color": "#B5482E",
            "fill-opacity": 0.08,
          },
        } as maplibregl.FillLayerSpecification,
        {
          id: `${id}--line`,
          type: "line",
          source: id,
          paint: {
            "line-color": "#B5482E",
            "line-width": 1,
            "line-opacity": 0.35,
            "line-dasharray": [2, 2],
          },
        } as maplibregl.LineLayerSpecification,
      ];
    case "facility_coverage":
      return [
        {
          id: `${id}--fill`,
          type: "fill",
          source: id,
          paint: {
            "fill-color": "#2E7D32",
            "fill-opacity": 0.06,
          },
        } as maplibregl.FillLayerSpecification,
        {
          id: `${id}--line`,
          type: "line",
          source: id,
          paint: {
            "line-color": "#2E7D32",
            "line-width": 1,
            "line-opacity": 0.4,
            "line-dasharray": [2, 2],
          },
        } as maplibregl.LineLayerSpecification,
      ];
    case "access_heatmap": {
      // Graduated colour green → amber → red on access_distance_m.
      const p95 = computeP95(
        (layer.geojson.features ?? []) as GeoJSON.Feature[],
        "access_distance_m",
      );
      const high = p95 && p95 > 0 ? p95 : 5000;
      const mid = high / 2;
      const colorExpr: maplibregl.DataDrivenPropertyValueSpecification<string> = [
        "interpolate",
        ["linear"],
        ["coalesce", ["get", "access_distance_m"], 0],
        0,
        "#2E7D32",
        mid,
        "#F9A825",
        high,
        "#C62828",
      ];
      return [
        {
          id: `${id}--circle`,
          type: "circle",
          source: id,
          paint: {
            "circle-color": colorExpr,
            "circle-radius": 4,
            "circle-opacity": 0.75,
            "circle-stroke-width": 0,
          },
        } as maplibregl.CircleLayerSpecification,
      ];
    }
    case "facility_gaps":
      return [
        {
          id: `${id}--circle`,
          type: "circle",
          source: id,
          paint: {
            "circle-color": "#C62828",
            "circle-radius": 7,
            "circle-opacity": 0.18,
            "circle-stroke-width": 2,
            "circle-stroke-color": "#C62828",
            "circle-stroke-opacity": 0.9,
          },
        } as maplibregl.CircleLayerSpecification,
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
): string[] {
  clearManagedLayers(map, managedSources);
  const interactiveLayerIds: string[] = [];
  for (const layer of sorted(layers)) {
    try {
      map.addSource(layer.id, {
        type: "geojson",
        data: layer.geojson,
      });
      managedSources.add(layer.id);
      for (const spec of layerSpecs(layer)) {
        map.addLayer(spec);
        // Everything that's a point/line benefits from hover tooltips.
        if (
          layer.role === "demand" ||
          layer.role === "candidate" ||
          layer.role === "selected" ||
          layer.role === "access_heatmap" ||
          layer.role === "facility_gaps"
        ) {
          interactiveLayerIds.push(spec.id);
        }
      }
    } catch (e) {
      console.warn("[MapView] failed to add layer", layer.id, e);
    }
  }
  return interactiveLayerIds;
}

// -----------------------------------------------------------------------
// Variant-aware metric curation
// -----------------------------------------------------------------------

interface CuratedMetric {
  label: string;
  value: string;
}

function curateMetrics(sol: MapSolution): CuratedMetric[] {
  const out: CuratedMetric[] = [];
  const m = sol.metrics ?? {};
  const get = (k: string): number | undefined => {
    const v = (m as Record<string, unknown>)[k];
    return typeof v === "number" && Number.isFinite(v) ? v : undefined;
  };

  const problem = (sol.problem_type ?? "").toLowerCase();

  // Universal top-line for every solver
  if (sol.objective_value != null) {
    out.push({
      label: "Objective",
      value: formatNumber(sol.objective_value),
    });
  }
  out.push({ label: "Facilities", value: String(sol.n_selected) });

  if (problem.includes("median")) {
    const avg = get("average_distance") ?? get("avg_distance");
    const total = get("total_weighted_distance") ?? get("total_distance");
    const maxD = get("max_distance");
    if (avg !== undefined) {
      out.push({ label: "Avg travel", value: formatDistanceMeters(avg) });
    }
    if (maxD !== undefined) {
      out.push({ label: "Max travel", value: formatDistanceMeters(maxD) });
    }
    if (total !== undefined) {
      out.push({
        label: "Total weighted",
        value: formatNumber(total),
      });
    }
  } else if (problem.includes("center")) {
    const worst = get("max_distance") ?? sol.objective_value ?? undefined;
    const avg = get("average_distance") ?? get("avg_distance");
    if (worst !== undefined) {
      out.push({ label: "Worst-case", value: formatDistanceMeters(worst) });
    }
    if (avg !== undefined) {
      out.push({ label: "Avg travel", value: formatDistanceMeters(avg) });
    }
  } else if (problem.includes("mclp")) {
    const cov =
      get("coverage_percentage") ??
      get("demand_covered_pct") ??
      get("coverage_pct");
    const covered = get("demand_covered") ?? get("covered_demand");
    if (cov !== undefined) {
      out.push({ label: "Coverage", value: `${cov.toFixed(1)}%` });
    }
    if (covered !== undefined) {
      out.push({ label: "Covered demand", value: formatNumber(covered) });
    }
    if (sol.service_radius_m) {
      out.push({
        label: "Service radius",
        value: formatDistanceMeters(sol.service_radius_m),
      });
    }
  } else if (problem.includes("lscp")) {
    const cov = get("coverage_percentage");
    if (cov !== undefined) {
      out.push({ label: "Coverage", value: `${cov.toFixed(1)}%` });
    }
    if (sol.service_radius_m) {
      out.push({
        label: "Service radius",
        value: formatDistanceMeters(sol.service_radius_m),
      });
    }
  } else {
    // Unknown variant: surface up to 3 numeric metrics as-is.
    Object.entries(m)
      .filter(
        ([, v]) => typeof v === "number" && Number.isFinite(v as number),
      )
      .slice(0, 3)
      .forEach(([k, v]) => {
        out.push({
          label: k.replace(/_/g, " "),
          value: formatNumber(v as number),
        });
      });
  }
  return out;
}

// -----------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------

const MIN_AOI_KM2 = 0.5;
const MAX_AOI_KM2 = 50_000;

interface Props {
  state: MapState;
}

interface HoverInfo {
  x: number;
  y: number;
  role: string;
  props: Record<string, unknown>;
}

export function MapView({ state }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const managedSources = useRef<Set<string>>(new Set());
  const interactiveLayerIds = useRef<string[]>([]);
  const loadedRef = useRef(false);
  const lastView = useRef({ lon: NaN, lat: NaN });
  const drawRef = useRef<MapboxDraw | null>(null);
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const [warningsOpen, setWarningsOpen] = useState(true);
  const [editingAoi, setEditingAoi] = useState(false);
  const [editAreaKm2, setEditAreaKm2] = useState<number | null>(null);
  const [editError, setEditError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const { refresh } = useSession();
  const queryClient = useQueryClient();
  const aoiName = useStore((s) => s.snapshot?.aoi?.name ?? "Custom area");

  const sol = state.solution;
  const warnings = useMemo(() => sol?.warnings ?? [], [sol]);

  const accessLayer = state.layers.find((l) => l.role === "access_heatmap");
  const accessP95 = useMemo(
    () =>
      accessLayer
        ? computeP95(
            (accessLayer.geojson.features ?? []) as GeoJSON.Feature[],
            "access_distance_m",
          )
        : null,
    [accessLayer],
  );

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
      interactiveLayerIds.current = applyLayers(
        map,
        state.layers,
        managedSources.current,
      );
    });

    const onMouseMove = (e: maplibregl.MapMouseEvent): void => {
      if (interactiveLayerIds.current.length === 0) {
        setHover(null);
        return;
      }
      const existing = interactiveLayerIds.current.filter((id) => {
        try {
          return !!map.getLayer(id);
        } catch {
          return false;
        }
      });
      if (existing.length === 0) {
        setHover(null);
        return;
      }
      const feats = map.queryRenderedFeatures(e.point, {
        layers: existing,
      });
      if (!feats || feats.length === 0) {
        setHover(null);
        map.getCanvas().style.cursor = "";
        return;
      }
      const top = feats[0];
      const layerId = top.layer.id;
      const role = layerId.includes("selected")
        ? "selected"
        : layerId.includes("candidate")
          ? "candidate"
          : layerId.includes("facility_gaps")
            ? "facility_gaps"
            : layerId.includes("access_heatmap")
              ? "access_heatmap"
              : "demand";
      setHover({
        x: e.point.x,
        y: e.point.y,
        role,
        props: (top.properties as Record<string, unknown>) ?? {},
      });
      map.getCanvas().style.cursor = "pointer";
    };
    const onMouseLeave = (): void => {
      setHover(null);
      map.getCanvas().style.cursor = "";
    };

    map.on("mousemove", onMouseMove);
    map.on("mouseout", onMouseLeave);

    return () => {
      loadedRef.current = false;
      map.off("mousemove", onMouseMove);
      map.off("mouseout", onMouseLeave);
      map.remove();
      mapRef.current = null;
      managedSources.current.clear();
      interactiveLayerIds.current = [];
    };
    // Run only on mount — subsequent state changes are handled below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync layers whenever state.layers changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;
    interactiveLayerIds.current = applyLayers(
      map,
      state.layers,
      managedSources.current,
    );
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

  // Re-open warnings banner whenever a new solution arrives.
  useEffect(() => {
    if (warnings.length > 0) setWarningsOpen(true);
  }, [warnings.length, sol?.status]);

  // ── Edit AOI handlers ─────────────────────────────────────────────────────

  const hasBoundary = state.layers.some((l) => l.role === "boundary");

  const onEditAoi = () => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;

    const draw = new MapboxDraw({
      displayControlsDefault: false,
      controls: { polygon: true, trash: true },
      styles: DRAW_STYLES,
    });
    map.addControl(draw as unknown as maplibregl.IControl);
    drawRef.current = draw;

    const boundaryLayer = state.layers.find((l) => l.role === "boundary");
    if (boundaryLayer && boundaryLayer.geojson.features.length > 0) {
      draw.add(boundaryLayer.geojson);

      // Compute initial area for display
      const m2 = area(boundaryLayer.geojson as Parameters<typeof area>[0]);
      const km2 = m2 / 1e6;
      setEditAreaKm2(km2);
      setEditError(null);
    }

    // Remove boundary layers from map visually while in edit mode
    const style = map.getStyle();
    for (const l of style?.layers ?? []) {
      if ((l as { source?: string }).source === "boundary_aoi") {
        try { map.removeLayer(l.id); } catch {}
      }
    }
    try { map.removeSource("boundary_aoi"); } catch {}
    managedSources.current.delete("boundary_aoi");

    const handleDrawChange = () => {
      const fc = draw.getAll();
      if (!fc.features.length) {
        setEditAreaKm2(null);
        setEditError(null);
        return;
      }
      const m2 = area(fc as Parameters<typeof area>[0]);
      const km2 = m2 / 1e6;
      if (km2 < MIN_AOI_KM2) {
        setEditAreaKm2(km2);
        setEditError(`Area too small (${km2.toFixed(3)} km²)`);
      } else if (km2 > MAX_AOI_KM2) {
        setEditAreaKm2(km2);
        setEditError(`Area too large (${Math.round(km2).toLocaleString()} km²)`);
      } else {
        setEditAreaKm2(km2);
        setEditError(null);
      }
    };

    map.on("draw.create", handleDrawChange);
    map.on("draw.update", handleDrawChange);
    map.on("draw.delete", handleDrawChange);

    setEditingAoi(true);
  };

  const onCancelEdit = () => {
    const map = mapRef.current;
    if (map && drawRef.current) {
      try { map.removeControl(drawRef.current as unknown as maplibregl.IControl); } catch {}
    }
    drawRef.current = null;
    setEditingAoi(false);
    setEditAreaKm2(null);
    setEditError(null);
    // Restore boundary layer
    if (map && loadedRef.current) {
      interactiveLayerIds.current = applyLayers(
        map,
        state.layers,
        managedSources.current,
      );
    }
  };

  const onSaveEdit = async () => {
    const map = mapRef.current;
    const draw = drawRef.current;
    if (!map || !draw || editError) return;

    const fc = draw.getAll();
    if (!fc.features.length) return;

    setSaving(true);
    try {
      await apiPost("/api/aoi/confirm", {
        name: aoiName,
        source: "search+refined",
        geojson: fc,
      });
      try { map.removeControl(draw as unknown as maplibregl.IControl); } catch {}
      drawRef.current = null;
      setEditingAoi(false);
      setEditAreaKm2(null);
      setEditError(null);
      refresh();
      await queryClient.invalidateQueries({ queryKey: ["map-state"] });
    } catch (e: unknown) {
      setEditError(e instanceof Error ? e.message : "Failed to save AOI.");
    } finally {
      setSaving(false);
    }
  };

  // ── Metrics / status ──────────────────────────────────────────────────────

  const curated = sol ? curateMetrics(sol) : [];
  const statusBadge = sol?.status
    ? sol.status === "optimal"
      ? { label: "Optimal", tone: "ok" as const }
      : sol.status === "feasible"
        ? { label: "Feasible", tone: "warn" as const }
        : sol.status === "timeout"
          ? { label: "Timed out", tone: "warn" as const }
          : sol.status === "ga_fallback"
            ? { label: "Heuristic", tone: "warn" as const }
            : { label: sol.status, tone: "muted" as const }
    : null;

  return (
    <div className="relative h-full w-full overflow-hidden">
      <div ref={containerRef} className="h-full w-full" />

      {/* Legend */}
      <div className="absolute bottom-8 left-3 z-10 flex flex-col gap-1 rounded border border-border bg-surface/90 px-2 py-1.5 text-2xs text-text-muted shadow-popover backdrop-blur">
        {state.layers.some((l) => l.role === "demand") && (
          <LegendRow color="#4285F4" label="Demand (by weight)" />
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
        {state.layers.some((l) => l.role === "coverage") && (
          <LegendRow color="#B5482E" label="Service radius" dashed />
        )}
        {accessLayer && <GradientLegendRow p95m={accessP95} />}
        {state.layers.some((l) => l.role === "facility_coverage") && (
          <LegendRow color="#2E7D32" label="Facility coverage" dashed />
        )}
        {state.layers.some((l) => l.role === "facility_gaps") && (
          <LegendRow color="#C62828" label="Uncovered (no nearby facility)" ring />
        )}
      </div>

      {/* Solution metrics overlay */}
      {sol && (
        <div className="absolute left-3 top-3 z-10 w-[260px] rounded border border-border bg-surface/90 px-3 py-2.5 text-xs text-text shadow-popover backdrop-blur">
          <div className="flex items-center justify-between gap-2">
            <div className="mono text-2xs font-medium uppercase tracking-wider text-text-faint">
              {sol.problem_type ?? "solution"}
              {sol.variant && sol.variant !== "base" && (
                <span className="ml-1 normal-case text-text-muted">
                  · {sol.variant}
                </span>
              )}
            </div>
            {statusBadge && (
              <span
                className={cn(
                  "rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide",
                  statusBadge.tone === "ok" &&
                    "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
                  statusBadge.tone === "warn" &&
                    "bg-amber-500/15 text-amber-700 dark:text-amber-400",
                  statusBadge.tone === "muted" &&
                    "bg-text-faint/10 text-text-muted",
                )}
              >
                {statusBadge.label}
              </span>
            )}
          </div>
          <div className="mt-2 space-y-0.5">
            {curated.map((row) => (
              <MetricRow key={row.label} label={row.label} value={row.value} />
            ))}
          </div>
          {(sol.solver || sol.solver_time_seconds || sol.distance_metric_used) && (
            <div className="mt-2 flex flex-wrap gap-1.5 border-t border-border pt-1.5 text-[10px] text-text-muted">
              {sol.solver && (
                <Tag>
                  solver · <span className="mono">{sol.solver}</span>
                </Tag>
              )}
              {typeof sol.solver_time_seconds === "number" && (
                <Tag>
                  <span className="mono">
                    {sol.solver_time_seconds.toFixed(1)}s
                  </span>
                </Tag>
              )}
              {sol.distance_metric_used && (
                <Tag>
                  distance · <span className="mono">{sol.distance_metric_used}</span>
                </Tag>
              )}
              {typeof sol.gap === "number" && (
                <Tag>
                  gap · <span className="mono">{(sol.gap * 100).toFixed(2)}%</span>
                </Tag>
              )}
            </div>
          )}
        </div>
      )}

      {/* Warnings banner */}
      {warnings.length > 0 && warningsOpen && (
        <div className="absolute right-3 top-3 z-10 w-[300px] rounded border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-900 shadow-popover backdrop-blur dark:text-amber-200">
          <div className="flex items-center justify-between gap-2">
            <span className="font-medium">Solver warnings ({warnings.length})</span>
            <button
              onClick={() => setWarningsOpen(false)}
              className="text-amber-900/70 hover:text-amber-900 dark:text-amber-200/70 dark:hover:text-amber-200"
              aria-label="Dismiss warnings"
            >
              ×
            </button>
          </div>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {warnings.slice(0, 5).map((w, i) => (
              <li key={i} className="leading-snug">
                {w}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Edit AOI button (idle state) */}
      {hasBoundary && !editingAoi && (
        <button
          onClick={onEditAoi}
          className="absolute bottom-2 left-3 z-10 flex items-center gap-1.5 rounded border border-border bg-surface/90 px-2 py-1 text-2xs text-text-muted shadow-popover backdrop-blur transition-colors hover:bg-surface-2 hover:text-text"
        >
          <Pencil className="h-2.5 w-2.5" strokeWidth={1.75} />
          Edit AOI
        </button>
      )}

      {/* Edit AOI overlay (active) */}
      {editingAoi && (
        <div className="absolute bottom-8 left-1/2 z-20 flex -translate-x-1/2 items-center gap-2.5 rounded-lg border border-border bg-surface/95 px-4 py-2.5 shadow-popover backdrop-blur">
          <span className="text-xs text-text-muted">Editing AOI boundary</span>
          {editAreaKm2 !== null && !editError && (
            <span className="mono rounded bg-accent/10 px-1.5 py-0.5 text-xs text-accent">
              {formatArea(editAreaKm2)}
            </span>
          )}
          {editError && (
            <span className="text-xs text-err">{editError}</span>
          )}
          <button
            onClick={onCancelEdit}
            className="flex items-center gap-1 rounded border border-border px-2 py-1 text-xs text-text-muted transition-colors hover:bg-surface-2 hover:text-text"
          >
            <X className="h-3 w-3" strokeWidth={2} />
            Cancel
          </button>
          <button
            onClick={onSaveEdit}
            disabled={!!editError || saving}
            className={cn(
              "flex items-center gap-1 rounded border px-2 py-1 text-xs font-medium transition-colors",
              editError || saving
                ? "cursor-not-allowed border-border text-text-faint"
                : "border-accent bg-accent text-white hover:opacity-90",
            )}
          >
            {saving ? (
              <Loader2 className="h-3 w-3 animate-spin" strokeWidth={1.75} />
            ) : (
              <Check className="h-3 w-3" strokeWidth={2} />
            )}
            Save
          </button>
        </div>
      )}

      {/* Hover tooltip */}
      {hover && <HoverTooltip info={hover} />}
    </div>
  );
}

// -----------------------------------------------------------------------
// Presentational helpers
// -----------------------------------------------------------------------

function LegendRow({
  color,
  label,
  line,
  dashed,
  ring,
}: {
  color: string;
  label: string;
  line?: boolean;
  dashed?: boolean;
  ring?: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      {line || dashed ? (
        <div
          className="h-px w-4 shrink-0"
          style={{
            background: dashed
              ? `repeating-linear-gradient(90deg, ${color} 0 3px, transparent 3px 6px)`
              : color,
          }}
        />
      ) : ring ? (
        <div
          className="h-2.5 w-2.5 shrink-0 rounded-full"
          style={{
            background: `${color}30`,
            border: `2px solid ${color}`,
          }}
        />
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

function GradientLegendRow({ p95m }: { p95m: number | null }) {
  const farLabel = p95m ? formatDistanceMeters(p95m) : "far";
  return (
    <div className="space-y-0.5">
      <span className="text-[10px] text-text-muted">Access distance</span>
      <div
        className="h-2 w-24 rounded-full"
        style={{
          background: "linear-gradient(to right, #2E7D32, #F9A825, #C62828)",
        }}
      />
      <div className="flex justify-between text-[9px] text-text-faint" style={{ width: 96 }}>
        <span>near</span>
        <span>{farLabel}</span>
      </div>
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className={cn("flex items-baseline justify-between gap-3")}>
      <span className="capitalize text-text-muted">{label}</span>
      <span className="mono text-text">{value}</span>
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded bg-text-faint/10 px-1.5 py-0.5">{children}</span>
  );
}

/** Lightweight hover card rendered in screen-space (not on the map). */
function HoverTooltip({ info }: { info: HoverInfo }) {
  const { x, y, role, props } = info;
  const rows: { label: string; value: string }[] = [];
  const pushNumeric = (label: string, key: string, fmt = formatNumber) => {
    const v = props[key];
    if (typeof v === "number" && Number.isFinite(v)) {
      rows.push({ label, value: fmt(v as number) });
    }
  };
  const name = typeof props.name === "string" ? props.name : null;
  if (role === "demand") {
    pushNumeric("Population", "population");
  } else if (role === "access_heatmap") {
    pushNumeric("Population", "population");
    pushNumeric("Distance to facility", "access_distance_m", formatDistanceMeters);
  } else if (role === "facility_gaps") {
    pushNumeric("Distance to nearest facility", "access_distance_m", formatDistanceMeters);
    const di = props.demand_idx;
    if (typeof di === "number") {
      rows.push({ label: "Demand #", value: String(di) });
    }
  } else if (role === "candidate" || role === "selected") {
    pushNumeric("Capacity", "capacity");
    pushNumeric("Cost", "cost");
    const fi = props.facility_idx;
    if (typeof fi === "number") {
      rows.push({ label: "Facility #", value: String(fi) });
    }
  }

  return (
    <div
      className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-[110%] rounded border border-border bg-surface/95 px-2 py-1.5 text-2xs text-text shadow-popover backdrop-blur"
      style={{ left: x, top: y }}
    >
      <div className="mono text-[10px] font-medium uppercase tracking-wider text-text-faint">
        {role === "access_heatmap"
          ? "demand point"
          : role === "facility_gaps"
            ? "uncovered demand"
            : role}
        {name ? ` · ${name}` : ""}
      </div>
      {rows.length === 0 ? (
        <div className="mt-0.5 text-text-muted">No attributes</div>
      ) : (
        <div className="mt-0.5 space-y-0.5">
          {rows.map((r) => (
            <div key={r.label} className="flex items-baseline justify-between gap-3">
              <span className="text-text-muted">{r.label}</span>
              <span className="mono">{r.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
