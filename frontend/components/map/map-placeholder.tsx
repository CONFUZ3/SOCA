"use client";

import { useStore } from "@/lib/store";
import { Chip } from "@/components/ui/chip";
import { formatArea, formatNumber } from "@/lib/format";
import { MapIcon } from "lucide-react";

/**
 * Temporary placeholder until the deck.gl + MapLibre canvas lands.
 * Intentionally quiet — does not pretend to be a real map.
 */
export function MapPlaceholder() {
  const snapshot = useStore((s) => s.snapshot);
  const datasets = useStore((s) => s.datasets);
  const aoi = snapshot?.aoi;

  return (
    <div className="relative flex h-full min-h-0 w-full flex-col overflow-hidden bg-mono-surface">
      {/* Subtle grid background — engineering blueprint vibe */}
      <div
        aria-hidden
        className="absolute inset-0 opacity-[0.45]"
        style={{
          backgroundImage:
            "linear-gradient(rgb(var(--border) / 1) 1px, transparent 1px), linear-gradient(90deg, rgb(var(--border) / 1) 1px, transparent 1px)",
          backgroundSize: "24px 24px",
        }}
      />

      <div className="relative z-10 flex-1 overflow-y-auto p-4">
        <div className="mx-auto max-w-2xl rounded-md border border-border bg-surface p-4">
          <div className="flex items-center gap-2 text-xs text-text-muted">
            <MapIcon className="h-3.5 w-3.5" strokeWidth={1.5} />
            Map canvas
          </div>
          <div className="mt-1 heading-section">
            {aoi ? aoi.name : "No area selected"}
          </div>
          {aoi ? (
            <div className="mt-1 flex items-center gap-2 text-xs text-text-muted">
              <Chip tone="accent">{formatArea(aoi.area_km2)}</Chip>
              <span className="mono text-2xs text-text-faint">
                source · {aoi.source}
              </span>
            </div>
          ) : null}

          <div className="mt-4 text-xs text-text-muted">
            The interactive map (deck.gl + MapLibre) lands in the next phase.
            For now, datasets and solutions are listed below as the agent
            loads them.
          </div>

          <div className="mt-4">
            <div className="text-2xs font-medium uppercase tracking-wider text-text-faint">
              Datasets in session
            </div>
            {datasets.length === 0 ? (
              <div className="mt-1 rounded border border-dashed border-border px-2 py-3 text-xs text-text-faint">
                None yet. Ask SOCA to fetch population, POIs, or upload a file.
              </div>
            ) : (
              <ul className="mt-1.5 divide-y divide-border rounded border border-border">
                {datasets.map((d) => (
                  <li
                    key={d.name}
                    className="flex items-center gap-2 px-2 py-1.5 text-xs"
                  >
                    <span className="mono truncate text-text">{d.name}</span>
                    <span className="ml-auto mono text-text-faint">
                      {formatNumber(d.num_features)} · {d.geometry_type}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
