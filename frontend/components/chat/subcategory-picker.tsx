"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/cn";
import { applySubcategoryFilter, nextToggled } from "@/lib/subcategory-filter";

export function SubcategoryPicker() {
  const datasets = useStore((s) => s.datasets);
  const qc = useQueryClient();

  const poiDatasets = datasets.filter(
    (d) => (d.available_subcategories?.length ?? 0) > 1,
  );

  if (poiDatasets.length === 0) return null;

  function toggle(datasetName: string, sub: string) {
    applySubcategoryFilter(datasetName, nextToggled(datasetName, sub)).then(() =>
      qc.invalidateQueries({ queryKey: ["map-state"] }),
    );
  }

  function setAll(datasetName: string, active: string[]) {
    applySubcategoryFilter(datasetName, active).then(() =>
      qc.invalidateQueries({ queryKey: ["map-state"] }),
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
