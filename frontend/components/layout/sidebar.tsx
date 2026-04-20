"use client";

import { Database, Download, FileText, Layers, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/cn";
import { formatNumber } from "@/lib/format";

export function Sidebar() {
  const datasets = useStore((s) => s.datasets);
  const snapshot = useStore((s) => s.snapshot);

  return (
    <aside className="flex h-full w-[260px] flex-col bg-surface hairline-r">
      <nav className="flex flex-col gap-0.5 p-2">
        <Rail icon={<Layers className="h-3.5 w-3.5" strokeWidth={1.5} />} label="Data" active />
        <Rail icon={<Database className="h-3.5 w-3.5" strokeWidth={1.5} />} label="Problem" />
        <Rail icon={<FileText className="h-3.5 w-3.5" strokeWidth={1.5} />} label="Reports" />
      </nav>

      <div className="hairline-t flex-1 overflow-y-auto px-3 pb-3 pt-3">
        <div className="heading-section">Datasets</div>
        {datasets.length === 0 ? (
          <div className="mt-2 rounded border border-dashed border-border px-2 py-3 text-xs text-text-faint">
            No datasets loaded. Ask SOCA or upload a file.
          </div>
        ) : (
          <ul className="mt-2 space-y-1">
            {datasets.map((d) => (
              <li
                key={d.name}
                className="rounded border border-border bg-bg px-2 py-1.5"
              >
                <div className="flex items-center gap-2">
                  <span className="mono truncate text-xs text-text">
                    {d.name}
                  </span>
                </div>
                <div className="mt-0.5 flex items-center gap-2 text-2xs text-text-faint">
                  <span className="mono">
                    {formatNumber(d.num_features)} · {d.geometry_type}
                  </span>
                  {d.source ? <span>· {d.source}</span> : null}
                </div>
              </li>
            ))}
          </ul>
        )}

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

        <div className="mt-5 heading-section">Exports</div>
        <div className="mt-2 flex flex-col gap-1">
          <ExportButton label="GeoJSON" href="/api/export/geojson" disabled />
          <ExportButton label="CSV" href="/api/export/csv" disabled />
          <ExportButton label="PDF report" href="/api/export/pdf" disabled />
        </div>
      </div>

      <div className="hairline-t px-3 py-2 text-2xs text-text-faint">
        <Sparkles className="mr-1 inline h-3 w-3" strokeWidth={1.5} />
        solver: gurobi ⇄ pulp
      </div>
    </aside>
  );
}

function Rail({
  icon,
  label,
  active,
}: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
}) {
  return (
    <button
      type="button"
      className={cn(
        "flex items-center gap-2 rounded px-2 py-1 text-xs text-text-muted",
        "hover:bg-surface-2 hover:text-text",
        active ? "bg-surface-2 text-text" : "",
      )}
    >
      <span className="text-text-faint">{icon}</span>
      {label}
    </button>
  );
}

function ExportButton({
  label,
  href,
  disabled,
}: {
  label: string;
  href: string;
  disabled?: boolean;
}) {
  const body = (
    <Button
      variant="subtle"
      size="sm"
      className="w-full justify-start"
      disabled={disabled}
    >
      <Download className="h-3.5 w-3.5" strokeWidth={1.5} />
      {label}
    </Button>
  );
  if (disabled) {
    return (
      <Tooltip content="Available after a solution is produced">{body}</Tooltip>
    );
  }
  return (
    <a href={href} target="_blank" rel="noreferrer">
      {body}
    </a>
  );
}
