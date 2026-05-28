"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useStore } from "@/lib/store";
import type { DatasetSummary } from "@/types";
import { cn } from "@/lib/cn";

async function applyFilter(
  dataset: DatasetSummary,
  next: string[],
  updateDatasetSubcategories: (name: string, active: string[]) => void,
  updateDatasetSummary: (summary: DatasetSummary) => void,
) {
  updateDatasetSubcategories(dataset.name, next);
  try {
    const resp = await fetch(
      `/api/data/${encodeURIComponent(dataset.name)}/filter`,
      {
        method: "PATCH",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active_subcategories: next }),
      },
    );
    if (resp.ok) {
      const body = (await resp.json()) as DatasetSummary;
      updateDatasetSummary(body);
    }
  } catch {
    // optimistic update stays; map resyncs on next poll
  }
}

export function SubcategoryPicker() {
  const datasets = useStore((s) => s.datasets);
  const updateDatasetSubcategories = useStore((s) => s.updateDatasetSubcategories);
  const updateDatasetSummary = useStore((s) => s.updateDatasetSummary);
  const qc = useQueryClient();

  const poiDatasets = datasets.filter(
    (d) => (d.available_subcategories?.length ?? 0) > 1,
  );

  if (poiDatasets.length === 0) return null;

  function toggle(datasetName: string, sub: string) {
    // Read the latest dataset from the store, not from the render closure.
    // Without this, two rapid clicks both see the same pre-click
    // `active_subcategories` and the second click reverts the first.
    const ds = useStore
      .getState()
      .datasets.find((d) => d.name === datasetName);
    if (!ds) return;
    const available = ds.available_subcategories ?? [];
    const current = ds.active_subcategories ?? available;
    const next = current.includes(sub)
      ? current.filter((s) => s !== sub)
      : [...current, sub];
    applyFilter(ds, next, updateDatasetSubcategories, updateDatasetSummary).then(
      () => qc.invalidateQueries({ queryKey: ["map-state"] }),
    );
  }

  function setAll(datasetName: string, active: string[]) {
    const ds = useStore
      .getState()
      .datasets.find((d) => d.name === datasetName);
    if (!ds) return;
    applyFilter(ds, active, updateDatasetSubcategories, updateDatasetSummary).then(
      () => qc.invalidateQueries({ queryKey: ["map-state"] }),
    );
  }

  return (
    <div className="flex flex-col gap-1.5 rounded border border-border bg-surface px-2.5 py-2 text-xs">
      <div className="flex items-center gap-1.5 text-text-muted">
        <span className="font-medium text-text">Filter facilities</span>
        <span className="text-text-faint">· select subtypes to include in analysis</span>
      </div>
      {poiDatasets.map((dataset) => {
        const available = dataset.available_subcategories ?? [];
        const active = new Set(dataset.active_subcategories ?? available);
        const allOn = active.size === available.length;

        return (
          <div key={dataset.name} className="flex flex-wrap items-center gap-1">
            <span className="shrink-0 text-[10px] text-text-faint">
              {dataset.name.replace(/_/g, " ")}:
            </span>
            {available.map((sub) => {
              const on = active.has(sub);
              return (
                <button
                  key={sub}
                  onClick={() => toggle(dataset.name, sub)}
                  className={cn(
                    "rounded-full border px-2 py-0.5 text-[10px] font-medium transition-colors",
                    on
                      ? "border-emerald-600/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                      : "border-border bg-surface-2 text-text-faint line-through",
                  )}
                >
                  {sub.replace(/_/g, " ")}
                </button>
              );
            })}
            <button
              onClick={() => setAll(dataset.name, allOn ? [] : available)}
              className="ml-auto shrink-0 text-[10px] text-text-faint hover:text-text"
            >
              {allOn ? "clear all" : "select all"}
            </button>
          </div>
        );
      })}
    </div>
  );
}
