"use client";

import { Download, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import { useStore } from "@/lib/store";
import { formatNumber } from "@/lib/format";
import { sourceLabel } from "@/lib/sources";
import { UploadDropzone } from "@/components/sidebar/upload-dropzone";

export function Sidebar() {
  const datasets = useStore((s) => s.datasets);
  const snapshot = useStore((s) => s.snapshot);
  const hasSolution = Boolean(snapshot?.has_solution);

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
                      {formatNumber(d.num_features)} · {d.geometry_type}
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

      <div className="hairline-t px-3 py-2 text-2xs text-text-faint">
        <Sparkles className="mr-1 inline h-3 w-3" strokeWidth={1.5} />
        solver: gurobi ⇄ pulp
      </div>
    </aside>
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
