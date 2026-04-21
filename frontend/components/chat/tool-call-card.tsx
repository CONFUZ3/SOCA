"use client";

import { useEffect, useState } from "react";
import {
  Atom,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Circle,
  Database,
  Loader2,
  MapPin,
  Sigma,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { formatDurationMs } from "@/lib/format";
import type { ChatTurnToolCall } from "@/lib/store";

/**
 * Readable label for each ADK tool.
 */
const LABELS: Record<string, { label: string; icon: React.ReactNode }> = {
  fetch_city_data: {
    label: "Fetching geographic data",
    icon: <MapPin className="h-3.5 w-3.5" strokeWidth={1.5} />,
  },
  stage_optimization: {
    label: "Staging optimization",
    icon: <Sigma className="h-3.5 w-3.5" strokeWidth={1.5} />,
  },
  confirm_optimization: {
    label: "Running solver",
    icon: <Atom className="h-3.5 w-3.5" strokeWidth={1.5} />,
  },
  get_data_status: {
    label: "Checking data status",
    icon: <Database className="h-3.5 w-3.5" strokeWidth={1.5} />,
  },
};

function statusGlyph(status: string) {
  switch (status) {
    case "ok":
      return <CheckCircle2 className="h-3 w-3 text-ok" strokeWidth={1.75} />;
    case "fail":
      return <XCircle className="h-3 w-3 text-err" strokeWidth={1.75} />;
    case "try":
      return <Loader2 className="h-3 w-3 animate-spin text-text-muted" strokeWidth={1.75} />;
    default:
      return <Circle className="h-3 w-3 text-text-faint" strokeWidth={1.5} />;
  }
}

export function ToolCallCard({ tc }: { tc: ChatTurnToolCall }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (tc.finishedAt) return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [tc.finishedAt]);

  const [open, setOpen] = useState(true);
  const meta = LABELS[tc.name] || {
    label: tc.name,
    icon: <Circle className="h-3.5 w-3.5" strokeWidth={1.5} />,
  };

  const elapsed = (tc.finishedAt ?? now) - tc.startedAt;
  const done = Boolean(tc.finishedAt);
  const failed =
    tc.summary &&
    typeof tc.summary === "object" &&
    (("status" in tc.summary && tc.summary.status === "error") ||
      ("staged" in tc.summary && tc.summary.staged === false));

  const caretRotate = open ? "rotate-90" : "";

  return (
    <div
      className={cn(
        "my-2 rounded border border-border bg-surface text-sm",
        "animate-slide-up",
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left"
      >
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-text-faint transition-transform duration-[120ms]",
            caretRotate,
          )}
          strokeWidth={1.75}
        />
        <span className="text-text-muted">{meta.icon}</span>
        <span className="flex-1 truncate">
          <span className="text-text">{meta.label}</span>
          <span className="ml-2 mono text-2xs text-text-faint">{tc.name}</span>
        </span>
        {done ? (
          failed ? (
            <XCircle className="h-3.5 w-3.5 text-err" strokeWidth={1.75} />
          ) : (
            <CheckCircle2 className="h-3.5 w-3.5 text-ok" strokeWidth={1.75} />
          )
        ) : (
          <Loader2
            className="h-3.5 w-3.5 animate-spin text-text-muted"
            strokeWidth={1.75}
          />
        )}
        <span className="mono text-2xs text-text-faint tabular-nums w-14 text-right">
          {formatDurationMs(elapsed)}
        </span>
      </button>

      {open ? (
        <div className="hairline-t px-2.5 py-1.5 text-xs text-text-muted">
          {/* Args line */}
          {Object.keys(tc.args).length > 0 ? (
            <div className="mono truncate text-2xs text-text-faint">
              {Object.entries(tc.args)
                .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                .join("  ")}
            </div>
          ) : null}

          {/* Nested activity events */}
          {tc.activity.length > 0 ? (
            <ul className="mt-1.5 space-y-0.5 border-l border-border pl-2">
              {tc.activity.map((evt, i) => (
                <li
                  key={i}
                  className="flex items-center gap-2 py-0.5 text-2xs"
                >
                  <span className="shrink-0">{statusGlyph(evt.status)}</span>
                  <span className="mono w-36 truncate text-text-muted">
                    {evt.stage}
                  </span>
                  {evt.source ? (
                    <span className="mono text-text-faint">
                      {evt.source}
                    </span>
                  ) : null}
                  <span className="flex-1 truncate text-text-muted">
                    {evt.detail}
                  </span>
                  {evt.duration_ms != null ? (
                    <span className="mono shrink-0 text-text-faint tabular-nums">
                      {formatDurationMs(evt.duration_ms)}
                    </span>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
