"use client";

import { useStore } from "@/lib/store";
import { cn } from "@/lib/cn";
import { formatDurationMs } from "@/lib/format";
import { CheckCircle2, Circle, Loader2, XCircle } from "lucide-react";

function glyph(status: string) {
  switch (status) {
    case "ok":
      return <CheckCircle2 className="h-3 w-3 text-ok" strokeWidth={1.75} />;
    case "fail":
      return <XCircle className="h-3 w-3 text-err" strokeWidth={1.75} />;
    case "try":
      return (
        <Loader2
          className="h-3 w-3 animate-spin text-text-muted"
          strokeWidth={1.75}
        />
      );
    default:
      return <Circle className="h-3 w-3 text-text-faint" strokeWidth={1.5} />;
  }
}

export function ActivityList() {
  const events = useStore((s) => s.activity);
  if (events.length === 0) {
    return (
      <div className="py-6 text-center text-xs text-text-faint">
        No activity yet.
      </div>
    );
  }
  return (
    <ul className="space-y-0.5 py-2">
      {events
        .slice()
        .reverse()
        .map((evt, i) => (
          <li
            key={i}
            className={cn(
              "flex items-center gap-2 rounded-sm px-1.5 py-1 text-2xs",
              "hover:bg-surface-2",
            )}
          >
            <span className="shrink-0">{glyph(evt.status)}</span>
            <span className="mono w-36 truncate text-text-muted">
              {evt.stage}
            </span>
            {evt.source ? (
              <span className="mono shrink-0 text-text-faint">
                {evt.source}
              </span>
            ) : null}
            <span className="flex-1 truncate text-text">{evt.detail}</span>
            {evt.duration_ms != null ? (
              <span className="mono shrink-0 tabular-nums text-text-faint">
                {formatDurationMs(evt.duration_ms)}
              </span>
            ) : null}
          </li>
        ))}
    </ul>
  );
}
