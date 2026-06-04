"use client";

import { useState } from "react";
import { Download, RotateCcw, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import { useStore } from "@/lib/store";
import { useSession } from "@/hooks/use-session";
import { formatNumber } from "@/lib/format";
import { sourceLabel } from "@/lib/sources";
import { UploadDropzone } from "@/components/sidebar/upload-dropzone";
import {
  applySubcategoryFilter,
  nextToggled,
} from "@/lib/subcategory-filter";
import type { DatasetSummary } from "@/types";

export function Sidebar() {
  const datasets = useStore((s) => s.datasets);
  const snapshot = useStore((s) => s.snapshot);
  const hasSolution = Boolean(snapshot?.has_solution);
  const { resetSession } = useSession();
  const [confirmReset, setConfirmReset] = useState(false);

  return (
    <aside className="flex h-full w-[260px] flex-col bg-surface hairline-r">
      <div className="flex-1 overflow-y-auto px-3 pb-3 pt-3">
        <div className="heading-section">Datasets</div>
        {datasets.length === 0 ? (
          <div className="mt-2 rounded border border-dashed border-border px-2 py-3 text-xs text-text-faint">
            No datasets loaded yet. Ask SOCA to fetch one, or upload below.
          </div>
        ) : (
          <ul className="mt-2 space-y-1">
            {datasets.map((d) => {
              const source = d.source || "local";
              const sourceDetails = d.source_details || [];
              const numericSummary =
                d.numeric_summary && d.numeric_summary.length > 0
                  ? d.numeric_summary.slice(0, 2)
                  : Object.entries(d.numeric_preview || {})
                      .slice(0, 2)
                      .map(([key, value]) => ({
                        key,
                        label: key,
                        value,
                      }));
              return (
                <li
                  key={d.name}
                  className="rounded border border-border bg-bg px-2 py-1.5"
                >
                  <div className="flex items-center gap-2">
                    <span className="mono truncate text-xs text-text">
                      {d.name}
                    </span>
                  </div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-2xs text-text-faint">
                    <span className="mono">
                      {d.active_num_features !== undefined &&
                      d.active_num_features !== d.num_features ? (
                        <>
                          <span className="text-accent">
                            {formatNumber(d.active_num_features)}
                          </span>
                          <span> / {formatNumber(d.num_features)}</span>
                        </>
                      ) : (
                        <>{formatNumber(d.num_features)}</>
                      )}
                      {" · "}
                      {d.geometry_type}
                    </span>
                  </div>
                  <Tooltip content={`Source: ${sourceLabel(source)}`}>
                    <div className="mt-0.5 flex items-center gap-1 text-2xs">
                      <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent/70" />
                      <span className="truncate text-text-muted">
                        {sourceLabel(source)}
                      </span>
                    </div>
                  </Tooltip>
                  <div className="mt-0.5 flex flex-wrap items-center gap-1 text-2xs text-text-faint">
                    {d.role ? (
                      <span className="rounded border border-border px-1 py-[1px] uppercase tracking-wide">
                        {d.role}
                      </span>
                    ) : null}
                    {sourceDetails.length > 0 ? (
                      <span className="truncate">
                        {sourceDetails.map((s) => sourceLabel(s)).join(" · ")}
                      </span>
                    ) : null}
                  </div>
                  {numericSummary.length > 0 ? (
                    <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-2xs text-text-faint">
                      {numericSummary.map((item) => (
                        <span key={item.key} className="mono">
                          {item.label}:{formatNumber(item.value)}
                        </span>
                      ))}
                    </div>
                  ) : null}
                  <SubcategoryFilter
                    dataset={d}
                    onToggle={(sub) =>
                      applySubcategoryFilter(d.name, nextToggled(d.name, sub))
                    }
                    onSetAll={(next) => applySubcategoryFilter(d.name, next)}
                  />
                </li>
              );
            })}
          </ul>
        )}

        <div className="mt-3">
          <UploadDropzone />
        </div>

        <div className="mt-5 heading-section">Problem</div>
        <div className="mt-2 rounded border border-border bg-bg p-2 text-xs text-text-muted">
          {snapshot?.problem_type ? (
            <>
              <div className="mono text-text">{snapshot.problem_type}</div>
              {snapshot.parameters &&
              Object.keys(snapshot.parameters).length > 0 ? (
                <ul className="mt-1.5 space-y-0.5">
                  {Object.entries(snapshot.parameters).map(([k, v]) => (
                    <li key={k} className="flex justify-between gap-2">
                      <span className="text-text-faint">{k}</span>
                      <span className="mono text-text">{String(v)}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
            </>
          ) : (
            <span className="text-text-faint">
              Awaiting parameters from the conversation.
            </span>
          )}
        </div>

        {hasSolution ? (
          <>
            <div className="mt-5 heading-section">Exports</div>
            <div className="mt-2 flex flex-col gap-1">
              <ExportButton label="GeoJSON" href="/api/export/geojson" />
              <ExportButton label="CSV (facilities)" href="/api/export/csv" />
              <ExportButton label="PDF report" href="/api/export/pdf" />
            </div>
          </>
        ) : null}
      </div>

      <div className="hairline-t px-3 py-2 flex items-center justify-between text-2xs text-text-faint">
        <span>
          <Sparkles className="mr-1 inline h-3 w-3" strokeWidth={1.5} />
          solver: gurobi ⇄ pulp
        </span>
        {confirmReset ? (
          <div className="flex items-center gap-1">
            <span className="text-text-muted">reset?</span>
            <button
              onClick={() => {
                setConfirmReset(false);
                resetSession();
              }}
              className="text-red-400 hover:text-red-300 transition-colors"
            >
              yes
            </button>
            <span>/</span>
            <button
              onClick={() => setConfirmReset(false)}
              className="hover:text-text transition-colors"
            >
              no
            </button>
          </div>
        ) : (
          <Tooltip content="Clear session and start over">
            <button
              onClick={() => setConfirmReset(true)}
              className="flex items-center gap-1 hover:text-text transition-colors"
            >
              <RotateCcw className="h-3 w-3" strokeWidth={1.5} />
              reset
            </button>
          </Tooltip>
        )}
      </div>
    </aside>
  );
}

function SubcategoryFilter({
  dataset,
  onToggle,
  onSetAll,
}: {
  dataset: DatasetSummary;
  onToggle: (sub: string) => void;
  onSetAll: (active: string[]) => void;
}) {
  const available = dataset.available_subcategories ?? [];
  if (available.length === 0) return null;

  const counts = dataset.subcategory_counts ?? {};
  const active = dataset.active_subcategories ?? available;
  const activeSet = new Set(active);
  const activeCount = active.length;
  const allOn = activeCount === available.length;
  const [expanded, setExpanded] = useState(true);

  const activeFeatureCount = available.reduce(
    (sum, sub) => (activeSet.has(sub) ? sum + (counts[sub] ?? 0) : sum),
    0,
  );
  const totalFeatureCount = available.reduce(
    (sum, sub) => sum + (counts[sub] ?? 0),
    0,
  );

  return (
    <div className="mt-1.5 border-t border-border/50 pt-1.5">
      <div className="flex items-center justify-between">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-1 text-2xs text-text-muted hover:text-text transition-colors"
        >
          <span className="font-medium">PoI subtypes</span>
          <span
            className={`mono ${activeCount < available.length ? "text-accent" : "text-text-faint"}`}
          >
            {activeCount}/{available.length}
          </span>
          <span className="text-text-faint">{expanded ? "▲" : "▼"}</span>
        </button>
        <div className="flex items-center gap-2 text-2xs text-text-faint">
          {totalFeatureCount > 0 && (
            <span className="mono">
              <span
                className={
                  activeFeatureCount < totalFeatureCount ? "text-accent" : ""
                }
              >
                {activeFeatureCount.toLocaleString()}
              </span>
              <span> / {totalFeatureCount.toLocaleString()} feat.</span>
            </span>
          )}
          {allOn ? (
            <button onClick={() => onSetAll([])} className="hover:text-text transition-colors">
              clear all
            </button>
          ) : (
            <button onClick={() => onSetAll(available)} className="hover:text-text transition-colors">
              select all
            </button>
          )}
        </div>
      </div>
      {expanded ? (
        <div className="mt-1 flex flex-col gap-0.5">
          {available.map((sub) => {
            const on = active.includes(sub);
            const n = counts[sub];
            return (
              <button
                key={sub}
                onClick={() => onToggle(sub)}
                title={on ? "Click to exclude from analysis" : "Click to include in analysis"}
                className={`flex items-center justify-between rounded border px-1.5 py-[3px] text-2xs transition-colors text-left ${
                  on
                    ? "border-accent/60 bg-accent/10 text-text"
                    : "border-border bg-transparent text-text-faint opacity-50"
                }`}
              >
                <span className={on ? "" : "line-through"}>{sub.replace(/_/g, " ")}</span>
                {n !== undefined && (
                  <span
                    className={`mono ml-2 shrink-0 ${
                      on ? "text-text-muted" : "text-text-faint"
                    }`}
                  >
                    {n.toLocaleString()}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function ExportButton({
  label,
  href,
}: {
  label: string;
  href: string;
}) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="block">
      <Button
        variant="subtle"
        size="sm"
        className="w-full justify-start"
      >
        <Download className="h-3.5 w-3.5" strokeWidth={1.5} />
        {label}
      </Button>
    </a>
  );
}
