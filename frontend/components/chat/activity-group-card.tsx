"use client";

import { useEffect, useRef, useState } from "react";
import {
  Activity,
  CheckCircle2,
  ChevronRight,
  Circle,
  Loader2,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { formatDurationMs } from "@/lib/format";
import { sourceLabel } from "@/lib/sources";
import {
  activityGroupStatus,
  formatActivityDetail,
  formatActivityHeadline,
  formatActivityStage,
  primaryActivityEvent,
} from "@/lib/activity-format";
import type { ActivityGroup } from "@/lib/store";

function statusGlyph(status: string) {
  switch (status) {
    case "ok":
      return <CheckCircle2 className="h-3 w-3 text-ok" strokeWidth={1.75} />;
    case "fail":
      return <XCircle className="h-3 w-3 text-err" strokeWidth={1.75} />;
    case "try":
      return (
        <Loader2 className="h-3 w-3 animate-spin text-text-muted" strokeWidth={1.75} />
      );
    default:
      return <Circle className="h-3 w-3 text-text-faint" strokeWidth={1.5} />;
  }
}

function summarise(group: ActivityGroup): {
  label: string;
  status: "try" | "ok" | "fail" | "info";
} {
  const events = group.events;
  if (events.length === 0) return { label: "Activity", status: "info" };
  const primary = primaryActivityEvent(events);
  const status = activityGroupStatus(events);
  const sources = Array.from(
    new Set(events.map((e) => e.source).filter((s): s is string => Boolean(s))),
  );
  const srcSuffix = sources.length > 0 ? ` · ${sources.map(sourceLabel).join(", ")}` : "";
  const head = primary ? formatActivityHeadline(primary, status) : "Activity updated";
  return { label: `${head}${srcSuffix}`, status };
}

export function ActivityGroupCard({ group }: { group: ActivityGroup }) {
  const { label, status } = summarise(group);
  // Auto-expand while in progress so the user can watch live steps;
  // auto-collapse on completion so finished groups don't clutter the chat.
  // Either action by the user (manual toggle) pins the state.
  const [open, setOpen] = useState(status === "try");
  const userTouched = useRef(false);
  useEffect(() => {
    if (userTouched.current) return;
    setOpen(status === "try");
  }, [status]);
  const totalMs = group.events.reduce(
    (sum, e) => sum + (e.duration_ms || 0),
    0,
  );

  return (
    <div className="my-2 rounded border border-border bg-surface/70 text-xs animate-slide-up">
      <button
        type="button"
        onClick={() => {
          userTouched.current = true;
          setOpen((o) => !o);
        }}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left"
      >
        <ChevronRight
          className={cn(
            "h-3 w-3 shrink-0 text-text-faint transition-transform duration-[120ms]",
            open ? "rotate-90" : "",
          )}
          strokeWidth={1.75}
        />
        <Activity className="h-3 w-3 shrink-0 text-text-faint" strokeWidth={1.5} />
        <span className="shrink-0">{statusGlyph(status)}</span>
        <span className="flex-1 truncate text-text-muted">{label}</span>
        <span className="mono text-2xs text-text-faint tabular-nums">
          {group.events.length} step{group.events.length === 1 ? "" : "s"}
        </span>
        {totalMs > 0 ? (
          <span className="mono text-2xs text-text-faint tabular-nums w-12 text-right">
            {formatDurationMs(totalMs)}
          </span>
        ) : null}
      </button>

      {open ? (
        <ul className="hairline-t space-y-0.5 px-2.5 py-1.5">
          {group.events.map((evt, i) => (
            <li
              key={i}
              className="flex items-center gap-2 py-0.5 text-2xs"
            >
              <span className="shrink-0">{statusGlyph(evt.status)}</span>
              <span className="mono w-32 shrink-0 truncate text-text-muted">
                {formatActivityStage(evt)}
              </span>
              {evt.source ? (
                <span className="mono shrink-0 text-text-faint">
                  {sourceLabel(evt.source)}
                </span>
              ) : null}
              <span className="flex-1 truncate text-text">{formatActivityDetail(evt)}</span>
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
  );
}
