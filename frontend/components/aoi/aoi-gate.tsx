"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { apiGet, apiPost, apiUpload, ApiError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Chip } from "@/components/ui/chip";
import { useSession } from "@/hooks/use-session";
import {
  ArrowRight,
  Loader2,
  MapPin,
  Pencil,
  Search,
  Upload,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { formatArea } from "@/lib/format";
import type { DatasetSummary, GeocodeCandidate } from "@/types";

// AoiDrawMap uses browser canvas APIs — skip SSR
const AoiDrawMap = dynamic(
  () => import("./aoi-draw-map").then((m) => m.AoiDrawMap),
  { ssr: false, loading: () => <div className="h-full w-full bg-mono-surface" /> },
);

interface ResolvedAoi {
  name: string;
  display_name: string;
  source: string;
  area_km2: number;
  geojson: unknown;
  // "boundary" = polygon upload used directly; "hull" = convex hull fitted to points/lines.
  derived?: "boundary" | "hull";
}

type Mode = "search" | "draw" | "upload";

export function AoiGate() {
  const [mode, setMode] = useState<Mode>("search");

  // ── Search state ──────────────────────────────────────────────────────────
  const [q, setQ] = useState("");
  const [suggestions, setSuggestions] = useState<GeocodeCandidate[]>([]);
  const [loadingSuggest, setLoadingSuggest] = useState(false);
  const [selected, setSelected] = useState<GeocodeCandidate | null>(null);
  const [resolved, setResolved] = useState<ResolvedAoi | null>(null);
  const [resolving, setResolving] = useState(false);

  // ── Draw state ────────────────────────────────────────────────────────────
  const [drawnGeojson, setDrawnGeojson] = useState<GeoJSON.FeatureCollection | null>(null);
  const [drawnAreaKm2, setDrawnAreaKm2] = useState<number | null>(null);
  const [drawError, setDrawError] = useState<string | null>(null);

  // ── Upload state ──────────────────────────────────────────────────────────
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  // The candidate AOI derived from the uploaded dataset (same shape as resolve).
  const [derived, setDerived] = useState<ResolvedAoi | null>(null);
  // Source points rendered (read-only) under the derived box for context.
  const [pointsFc, setPointsFc] = useState<GeoJSON.FeatureCollection | null>(null);
  // All geometry-bearing datasets from this upload (any can define the AOI;
  // the rest are kept in the session for the optimization step).
  const [uploadedDs, setUploadedDs] = useState<DatasetSummary[]>([]);
  // Name of the dataset currently defining the AOI.
  const [aoiSource, setAoiSource] = useState<string | null>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);

  // ── Shared ────────────────────────────────────────────────────────────────
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { refresh } = useSession();

  const debounced = useDebouncedValue(q, 220);

  useEffect(() => {
    if (!debounced || debounced.trim().length < 3) {
      setSuggestions([]);
      return;
    }
    let cancelled = false;
    setLoadingSuggest(true);
    apiGet<{ candidates: GeocodeCandidate[] }>(
      `/api/aoi/suggest?q=${encodeURIComponent(debounced.trim())}&limit=6`,
    )
      .then((r) => {
        if (!cancelled) setSuggestions(r.candidates || []);
      })
      .catch(() => {
        if (!cancelled) setSuggestions([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingSuggest(false);
      });
    return () => {
      cancelled = true;
    };
  }, [debounced]);

  const onPick = async (c: GeocodeCandidate) => {
    setSelected(c);
    setResolved(null);
    setError(null);
    setResolving(true);
    try {
      const r = await apiPost<ResolvedAoi>("/api/aoi/resolve", {
        candidate: c,
      });
      setResolved(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to resolve boundary.");
    } finally {
      setResolving(false);
    }
  };

  const onConfirmSearch = async () => {
    if (!resolved) return;
    setConfirming(true);
    setError(null);
    try {
      await apiPost("/api/aoi/confirm", {
        name: resolved.name,
        source: resolved.source,
        geojson: resolved.geojson,
      });
      refresh();
    } catch (e: unknown) {
      setError(
        e instanceof Error ? e.message : "Failed to confirm area of interest.",
      );
    } finally {
      setConfirming(false);
    }
  };

  const onConfirmDraw = async () => {
    if (!drawnGeojson || drawError) return;
    setConfirming(true);
    setError(null);
    try {
      await apiPost("/api/aoi/confirm", {
        name: "Custom drawn area",
        source: "drawn",
        geojson: drawnGeojson,
      });
      refresh();
    } catch (e: unknown) {
      setError(
        e instanceof Error ? e.message : "Failed to confirm area of interest.",
      );
    } finally {
      setConfirming(false);
    }
  };

  const handleDrawn = (fc: GeoJSON.FeatureCollection, km2: number) => {
    setDrawnGeojson(fc);
    setDrawnAreaKm2(km2);
  };

  const handleAreaChange = (km2: number | null, err: string | null) => {
    setDrawnAreaKm2(km2);
    setDrawError(err);
    if (err || km2 === null) setDrawnGeojson(null);
  };

  // Clear transient drawing/upload state when switching tabs so geometry from
  // one mode never leaks into another mode's confirm payload.
  const resetTransient = () => {
    setDrawnGeojson(null);
    setDrawnAreaKm2(null);
    setDrawError(null);
    setDerived(null);
    setPointsFc(null);
    setUploadedDs([]);
    setAoiSource(null);
    setUploadError(null);
    setError(null);
  };

  const switchMode = (m: Mode) => {
    resetTransient();
    setMode(m);
  };

  // Derive the AOI from a named dataset already loaded in the session, and
  // load its geometry as the read-only overlay. Used on first upload and when
  // the user switches which dataset should define the AOI.
  const deriveFrom = async (name: string) => {
    setDrawnGeojson(null);
    setDrawnAreaKm2(null);
    setDrawError(null);
    setError(null);
    const r = await apiPost<ResolvedAoi>("/api/aoi/from-dataset", { name });
    // Fetch the source geometry BEFORE flipping to the preview so the overlay
    // is present when AoiDrawMap (re)mounts — it reads pointsGeojson once.
    // The endpoint serves "application/geo+json", which apiGet returns as raw
    // text, so parse it back into an object when needed.
    try {
      const raw = await apiGet<unknown>(
        `/api/data/${encodeURIComponent(name)}.geojson`,
      );
      const fc = typeof raw === "string" ? JSON.parse(raw) : raw;
      setPointsFc(fc as GeoJSON.FeatureCollection);
    } catch {
      setPointsFc(null);
    }
    setAoiSource(name);
    setDerived(r);
  };

  const onUploadFiles = async (files: FileList | File[] | null) => {
    const list = files ? Array.from(files) : [];
    if (!list.length) return;
    setUploadBusy(true);
    setUploadError(null);
    setError(null);
    try {
      // Every parsed file is stored in the session; the non-AOI ones stay
      // available for the optimization step (sidebar datasets).
      const res = await apiUpload<{
        loaded: DatasetSummary[];
        errors: Array<{ name: string; error: string }>;
      }>("/api/data/upload", list);
      if (res.errors?.length) {
        setUploadError(res.errors.map((e) => `${e.name}: ${e.error}`).join("; "));
      }
      const loaded = res.loaded || [];
      const geoDs = loaded.filter((d) =>
        /point|line|polygon/i.test(d.geometry_type),
      );
      setUploadedDs(geoDs);
      // Prefer a polygon (used as the AOI directly); else points/lines get a
      // fitted bounding box server-side.
      const pick =
        geoDs.find((d) => /polygon/i.test(d.geometry_type)) ??
        geoDs.find((d) => /point|line/i.test(d.geometry_type)) ??
        loaded[0];
      if (!pick) {
        if (!res.errors?.length) setUploadError("No usable geometry in upload.");
        return;
      }
      await deriveFrom(pick.name);
    } catch (e: unknown) {
      const msg =
        e instanceof ApiError
          ? typeof e.detail === "string"
            ? e.detail
            : e.message
          : e instanceof Error
            ? e.message
            : "Upload failed.";
      setUploadError(msg);
    } finally {
      setUploadBusy(false);
    }
  };

  const onConfirmUpload = async () => {
    if (!derived || drawError) return;
    setConfirming(true);
    setError(null);
    try {
      await apiPost("/api/aoi/confirm", {
        name: derived.name,
        source: "upload",
        // Use edited geometry if the user dragged vertices; otherwise the
        // unedited derived box (no draw.update fires until an edit).
        geojson: drawnGeojson ?? derived.geojson,
      });
      refresh();
    } catch (e: unknown) {
      setError(
        e instanceof Error ? e.message : "Failed to confirm area of interest.",
      );
    } finally {
      setConfirming(false);
    }
  };

  const tabToggle = (
    <div className="flex items-center gap-1 rounded-md border border-border bg-surface p-0.5">
      <button
        type="button"
        onClick={() => switchMode("search")}
        className={cn(
          "flex items-center gap-1.5 rounded px-2.5 py-1 text-xs font-medium transition-colors",
          mode === "search"
            ? "bg-surface-2 text-text shadow-sm"
            : "text-text-muted hover:text-text",
        )}
      >
        <Search className="h-3 w-3" strokeWidth={1.75} />
        Search
      </button>
      <button
        type="button"
        onClick={() => switchMode("draw")}
        className={cn(
          "flex items-center gap-1.5 rounded px-2.5 py-1 text-xs font-medium transition-colors",
          mode === "draw"
            ? "bg-surface-2 text-text shadow-sm"
            : "text-text-muted hover:text-text",
        )}
      >
        <Pencil className="h-3 w-3" strokeWidth={1.75} />
        Draw
      </button>
      <button
        type="button"
        onClick={() => switchMode("upload")}
        className={cn(
          "flex items-center gap-1.5 rounded px-2.5 py-1 text-xs font-medium transition-colors",
          mode === "upload"
            ? "bg-surface-2 text-text shadow-sm"
            : "text-text-muted hover:text-text",
        )}
      >
        <Upload className="h-3 w-3" strokeWidth={1.75} />
        Upload
      </button>
    </div>
  );

  // Shared full-screen editable-map layout used by Draw mode and the Upload
  // preview (derived bounds). AoiDrawMap drives handleDrawn/handleAreaChange.
  const editableMapView = (params: {
    instructions: string;
    initialGeojson?: GeoJSON.FeatureCollection | null;
    pointsGeojson?: GeoJSON.FeatureCollection | null;
    area: number | null;
    confirmDisabled: boolean;
    emptyHint: string | null;
    onConfirm: () => void;
    extra?: ReactNode;
    // Changing this remounts the map so a new initialGeojson/overlay loads
    // (AoiDrawMap reads those once, on mount).
    mapKey?: string;
  }) => (
    <div className="relative h-full w-full">
      <AoiDrawMap
        key={params.mapKey}
        onDrawn={handleDrawn}
        onAreaChange={handleAreaChange}
        initialGeojson={params.initialGeojson}
        pointsGeojson={params.pointsGeojson}
      />

      {/* Top-center overlay: tab toggle + instructions */}
      <div className="pointer-events-none absolute left-1/2 top-4 z-10 flex -translate-x-1/2 flex-col items-center gap-2">
        <div className="pointer-events-auto">{tabToggle}</div>
        {params.extra ? (
          <div className="pointer-events-auto">{params.extra}</div>
        ) : null}
        <div className="rounded border border-border bg-surface/90 px-3 py-1.5 text-xs text-text-muted backdrop-blur">
          {params.instructions}
        </div>
      </div>

      {/* Bottom bar: area + confirm */}
      <div className="absolute bottom-8 left-1/2 z-10 flex -translate-x-1/2 items-center gap-2.5 rounded-lg border border-border bg-surface/95 px-4 py-2.5 shadow-popover backdrop-blur">
        {params.area !== null && !drawError && (
          <Chip tone="accent">{formatArea(params.area)}</Chip>
        )}
        {drawError ? (
          <span className="text-xs text-err">{drawError}</span>
        ) : params.emptyHint ? (
          <span className="text-xs text-text-muted">{params.emptyHint}</span>
        ) : null}
        <Button
          variant="primary"
          size="md"
          onClick={params.onConfirm}
          disabled={params.confirmDisabled}
        >
          {confirming ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.75} />
          ) : (
            <>
              Confirm area
              <ArrowRight className="h-3.5 w-3.5" strokeWidth={2} />
            </>
          )}
        </Button>
      </div>

      {error && (
        <div className="absolute bottom-24 left-1/2 z-10 -translate-x-1/2 rounded border border-err/30 bg-surface/95 px-2.5 py-1.5 text-xs text-err backdrop-blur">
          {error}
        </div>
      )}
    </div>
  );

  // ── Draw mode layout ───────────────────────────────────────────────────────
  if (mode === "draw") {
    return editableMapView({
      instructions: "Click to place vertices · double-click to close the polygon",
      area: drawnAreaKm2,
      confirmDisabled: !drawnGeojson || !!drawError || confirming,
      emptyHint:
        drawnGeojson === null ? "Draw a polygon on the map to continue" : null,
      onConfirm: onConfirmDraw,
    });
  }

  // ── Upload mode layout ──────────────────────────────────────────────────────
  if (mode === "upload") {
    // Phase 2 — bounds derived from the upload: editable preview + confirm.
    if (derived) {
      return editableMapView({
        instructions:
          derived.derived === "boundary"
            ? "Uploaded polygon · drag vertices to refine, or confirm as-is"
            : "Outline fitted to your data · drag vertices to refine, or confirm as-is",
        initialGeojson: derived.geojson as GeoJSON.FeatureCollection,
        pointsGeojson: pointsFc,
        area: drawnAreaKm2 ?? derived.area_km2,
        confirmDisabled: !!drawError || confirming,
        emptyHint: null,
        onConfirm: onConfirmUpload,
        mapKey: aoiSource ?? "derived",
        extra: (
          <div className="flex flex-col items-center gap-1.5">
            {uploadedDs.length > 1 ? (
              <div className="flex flex-wrap items-center justify-center gap-1 rounded border border-border bg-surface/90 px-2 py-1 backdrop-blur">
                <span className="text-2xs text-text-faint">AOI from</span>
                {uploadedDs.map((d) => (
                  <button
                    key={d.name}
                    type="button"
                    onClick={() => void deriveFrom(d.name)}
                    className={cn(
                      "rounded px-1.5 py-0.5 text-2xs transition-colors",
                      d.name === aoiSource
                        ? "bg-accent text-white"
                        : "text-text-muted hover:bg-surface-2",
                    )}
                  >
                    {d.name}
                  </button>
                ))}
              </div>
            ) : null}
            <button
              type="button"
              onClick={resetTransient}
              className="rounded border border-border bg-surface/90 px-2.5 py-1 text-xs text-text-muted backdrop-blur hover:text-text"
            >
              ← Upload different files
            </button>
            {uploadedDs.length > 1 ? (
              <div className="rounded bg-surface/90 px-2 py-0.5 text-2xs text-text-faint backdrop-blur">
                {uploadedDs.length} datasets loaded · the rest are kept for analysis
              </div>
            ) : null}
          </div>
        ),
      });
    }

    // Phase 1 — upload a dataset.
    return (
      <div className="flex h-full w-full items-center justify-center bg-bg px-6 py-10">
        <div className="mx-auto w-full max-w-xl">
          <div className="mb-5 flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="text-2xs font-medium uppercase tracking-wider text-text-faint">
                Step 1 of 2
              </div>
              <h1 className="heading-panel mt-1">Upload data to set the area</h1>
              <p className="mt-1.5 text-sm text-text-muted">
                Upload one or more datasets. A polygon defines the area
                directly; points or lines get a bounding box fitted around them.
                Any extra files are kept for the optimization step.
              </p>
            </div>
            <div className="shrink-0 pt-0.5">{tabToggle}</div>
          </div>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              void onUploadFiles(e.dataTransfer.files);
            }}
            onClick={() => uploadInputRef.current?.click()}
            role="button"
            tabIndex={0}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-md border border-dashed px-4 py-10 text-sm transition-colors",
              dragOver
                ? "border-accent bg-surface-2 text-text"
                : "border-border bg-surface text-text-muted hover:border-accent/60 hover:text-text",
            )}
          >
            <input
              ref={uploadInputRef}
              type="file"
              multiple
              accept=".geojson,.json,.zip,.csv,.shp"
              className="hidden"
              onChange={(e) => {
                void onUploadFiles(e.target.files);
                if (uploadInputRef.current) uploadInputRef.current.value = "";
              }}
            />
            {uploadBusy ? (
              <Loader2 className="h-5 w-5 animate-spin" strokeWidth={1.5} />
            ) : (
              <Upload className="h-5 w-5" strokeWidth={1.5} />
            )}
            <span>
              {uploadBusy
                ? "Reading & deriving bounds…"
                : dragOver
                  ? "Drop to upload"
                  : "Drag & drop or click to upload"}
            </span>
            <span className="text-2xs text-text-faint">
              GeoJSON, CSV, SHP.zip · points, lines, or polygons
            </span>
          </div>

          {uploadError ? (
            <div className="mt-3 flex items-center gap-1.5 rounded border border-err/30 bg-accent-soft px-2.5 py-1.5 text-xs text-err">
              <XCircle className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
              <span>{uploadError}</span>
            </div>
          ) : null}
        </div>
      </div>
    );
  }

  // ── Search mode layout (default) ──────────────────────────────────────────
  return (
    <div className="flex h-full w-full items-center justify-center bg-bg px-6 py-10">
      <div className="mx-auto w-full max-w-xl">
        <div className="mb-5 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="text-2xs font-medium uppercase tracking-wider text-text-faint">
              Step 1 of 2
            </div>
            <h1 className="heading-panel mt-1">Pick an area of interest</h1>
            <p className="mt-1.5 text-sm text-text-muted">
              Search for a place or switch to Draw to sketch your own boundary.
            </p>
          </div>
          <div className="shrink-0 pt-0.5">{tabToggle}</div>
        </div>

        <div className="relative">
          <Search
            className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-faint"
            strokeWidth={1.5}
          />
          <Input
            className="h-10 pl-7 text-md"
            placeholder="e.g. Lima, Peru"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            autoFocus
          />
          {loadingSuggest ? (
            <Loader2
              className="absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 animate-spin text-text-faint"
              strokeWidth={1.75}
            />
          ) : null}
        </div>

        {suggestions.length > 0 && !resolved ? (
          <ul className="mt-2 overflow-hidden rounded-md border border-border bg-surface">
            {suggestions.map((c, i) => (
              <li key={`${c.display_name}-${i}`}>
                <button
                  type="button"
                  onClick={() => onPick(c)}
                  className={cn(
                    "flex w-full items-center gap-2 border-b border-border px-2.5 py-2 text-left text-sm last:border-b-0",
                    "transition-colors hover:bg-surface-2",
                    selected?.display_name === c.display_name && "bg-surface-2",
                  )}
                >
                  <MapPin
                    className="h-3.5 w-3.5 shrink-0 text-text-faint"
                    strokeWidth={1.5}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2">
                      <span className="truncate text-text">{c.short_name}</span>
                      {c.kind ? (
                        <span className="mono text-2xs text-text-faint">
                          {c.kind}
                        </span>
                      ) : null}
                    </div>
                    {c.context ? (
                      <div className="truncate text-xs text-text-muted">
                        {c.context}
                      </div>
                    ) : null}
                  </div>
                  {c.has_relation ? (
                    <Chip tone="ok">real boundary</Chip>
                  ) : (
                    <Chip tone="muted">bbox</Chip>
                  )}
                </button>
              </li>
            ))}
          </ul>
        ) : null}

        {resolving ? (
          <div className="mt-3 flex items-center gap-2 text-xs text-text-muted">
            <Loader2 className="h-3 w-3 animate-spin" strokeWidth={1.75} />
            Fetching boundary polygon from OpenStreetMap…
          </div>
        ) : null}

        {resolved ? (
          <div className="mt-3 rounded-md border border-border bg-surface p-3">
            <div className="flex items-start gap-2">
              <MapPin
                className="mt-0.5 h-4 w-4 shrink-0 text-accent"
                strokeWidth={1.75}
              />
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-text">
                  {resolved.name}
                </div>
                <div className="truncate text-xs text-text-muted">
                  {resolved.display_name}
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <Chip tone="accent">{formatArea(resolved.area_km2)}</Chip>
                  <Chip tone="muted">source · {resolved.source}</Chip>
                </div>
              </div>
            </div>
            <div className="mt-3 flex items-center justify-end gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setResolved(null);
                  setSelected(null);
                }}
              >
                Back
              </Button>
              <Button
                variant="primary"
                size="md"
                onClick={onConfirmSearch}
                disabled={confirming}
              >
                Confirm area
                <ArrowRight className="h-3.5 w-3.5" strokeWidth={2} />
              </Button>
            </div>
          </div>
        ) : null}

        {error ? (
          <div className="mt-3 rounded border border-err/30 bg-accent-soft px-2.5 py-1.5 text-xs text-err">
            {error}
          </div>
        ) : null}

        <div className="mt-6 flex flex-wrap items-center gap-1.5 text-2xs text-text-faint">
          <span className="mono">Try</span>
          {["Lima, Peru", "Brooklyn, New York", "Nairobi, Kenya", "Mirpur, Dhaka"].map(
            (ex) => (
              <button
                key={ex}
                type="button"
                className="rounded-sm border border-border bg-surface px-1.5 py-0.5 text-text-muted hover:bg-surface-2"
                onClick={() => setQ(ex)}
              >
                {ex}
              </button>
            ),
          )}
        </div>
      </div>
    </div>
  );
}

function useDebouncedValue<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  const ref = useRef<number | null>(null);
  useEffect(() => {
    if (ref.current) window.clearTimeout(ref.current);
    ref.current = window.setTimeout(() => setV(value), ms);
    return () => {
      if (ref.current) window.clearTimeout(ref.current);
    };
  }, [value, ms]);
  return v;
}
