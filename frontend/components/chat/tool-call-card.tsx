"use client";

import { useEffect, useState } from "react";
import {
  Atom,
  CheckCircle2,
  ChevronRight,
  Circle,
  Database,
  Loader2,
  MapPin,
  Sigma,
  TriangleAlert,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { formatDurationMs, formatNumber } from "@/lib/format";
import { sourceLabel } from "@/lib/sources";
import type { ChatTurnToolCall } from "@/lib/store";

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

function prettyArgs(name: string, args: Record<string, unknown>): string | null {
  if (!args || Object.keys(args).length === 0) return null;
  if (name === "fetch_city_data") {
    const loc = args.location as string | undefined;
    const scale = args.scale as string | undefined;
    const parts = [loc, scale].filter(Boolean);
    return parts.length ? parts.join(" · ") : null;
  }
  if (name === "stage_optimization") {
    const pt = args.problem_type as string | undefined;
    const n = args.n_facilities as number | undefined;
    const radius = args.service_radius as number | string | undefined;
    const variant = args.variant as string | undefined;
    const parts = [
      pt,
      variant && variant !== "base" ? variant : null,
      n != null ? `${n} facilities` : null,
      radius != null ? `radius ${radius}` : null,
    ].filter(Boolean);
    return parts.join(" · ") || null;
  }
  if (name === "confirm_optimization") return null; // usually empty / noisy
  return Object.entries(args)
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join("  ");
}

type Summary = Record<string, unknown>;

function Row({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="flex items-baseline gap-2 py-0.5 text-2xs">
      <span className="w-20 shrink-0 text-text-faint">{label}</span>
      <span className="flex-1 text-text">{value}</span>
    </div>
  );
}

function FetchCityDataSummary({ summary }: { summary: Summary }) {
  const status = String(summary.status || "");
  const fetched = Array.isArray(summary.fetched_datasets)
    ? (summary.fetched_datasets as string[])
    : [];
  const summaries = Array.isArray(summary.summaries)
    ? (summary.summaries as string[])
    : [];
  const errors = Array.isArray(summary.errors)
    ? (summary.errors as Array<Record<string, unknown>>)
    : [];

  return (
    <div className="mt-1 space-y-0.5">
      <Row
        label="Result"
        value={
          <span
            className={cn(
              "mono",
              status === "success"
                ? "text-ok"
                : status === "partial"
                  ? "text-warn"
                  : "text-err",
            )}
          >
            {status || "unknown"} · {fetched.length} dataset
            {fetched.length === 1 ? "" : "s"}
          </span>
        }
      />
      {summaries.length > 0 ? (
        <Row
          label="Steps"
          value={
            <ul className="space-y-0.5">
              {summaries.map((s, i) => (
                <li key={i} className="text-text-muted">
                  {s}
                </li>
              ))}
            </ul>
          }
        />
      ) : null}
      {errors.length > 0 ? (
        <Row
          label="Errors"
          value={
            <ul className="space-y-0.5 text-err">
              {errors.map((e, i) => (
                <li key={i}>
                  {(e.step as string) || "step"}: {(e.error as string) || "failed"}
                </li>
              ))}
            </ul>
          }
        />
      ) : null}
    </div>
  );
}

function StageOptimizationSummary({ summary }: { summary: Summary }) {
  const staged = summary.staged === true;
  const pt = summary.problem_type as string | undefined;
  const params = (summary.parameters || {}) as Record<string, unknown>;
  const dataStatus = (summary.data_status || {}) as Record<string, unknown>;
  const warnings = Array.isArray(summary.validation_warnings)
    ? (summary.validation_warnings as string[])
    : [];
  const error = summary.error as string | undefined;

  if (!staged) {
    return (
      <div className="mt-1 space-y-0.5">
        <Row
          label="Status"
          value={<span className="text-err">Not staged</span>}
        />
        {error ? <Row label="Reason" value={error} /> : null}
      </div>
    );
  }

  const demand = (dataStatus.demand_datasets as string[]) || [];
  const candidate = (dataStatus.candidate_datasets as string[]) || [];
  const boundary = (dataStatus.boundary_datasets as string[]) || [];
  const willGenerate = Boolean(dataStatus.will_generate_candidates);

  return (
    <div className="mt-1 space-y-0.5">
      <Row
        label="Problem"
        value={
          <span className="mono text-text">
            {pt || "unknown"}
            {params.variant && params.variant !== "base"
              ? ` · ${params.variant}`
              : ""}
          </span>
        }
      />
      {params.n_facilities != null ? (
        <Row label="Facilities" value={<span className="mono">{String(params.n_facilities)}</span>} />
      ) : null}
      {params.service_radius != null ? (
        <Row
          label="Radius"
          value={<span className="mono">{String(params.service_radius)} m</span>}
        />
      ) : null}
      <Row
        label="Distance"
        value={<span className="mono">{String(params.distance_metric || "network")}</span>}
      />
      <Row
        label="Data"
        value={
          <span className="text-text-muted">
            {boundary.length} boundary · {demand.length} demand ·{" "}
            {candidate.length} candidate
            {willGenerate ? " (will generate)" : ""}
          </span>
        }
      />
      {warnings.length > 0 ? (
        <Row
          label="Warnings"
          value={
            <ul className="space-y-0.5 text-warn">
              {warnings.map((w, i) => (
                <li key={i} className="flex items-start gap-1">
                  <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" strokeWidth={1.75} />
                  <span>{w}</span>
                </li>
              ))}
            </ul>
          }
        />
      ) : null}
    </div>
  );
}

function ConfirmOptimizationSummary({ summary }: { summary: Summary }) {
  const status = String(summary.status || "");
  const obj = summary.objective_value as number | null | undefined;
  const n = summary.num_facilities_selected as number | null | undefined;
  const metric = summary.distance_metric_used as string | undefined;
  const error = summary.error_message as string | undefined;
  const warnings = Array.isArray(summary.warnings)
    ? (summary.warnings as string[])
    : [];

  const ok = status === "optimal" || status === "feasible" || status === "success";

  return (
    <div className="mt-1 space-y-0.5">
      <Row
        label="Status"
        value={
          <span
            className={cn(
              "mono",
              ok ? "text-ok" : status === "timeout" ? "text-warn" : "text-err",
            )}
          >
            {status || "unknown"}
          </span>
        }
      />
      {obj != null ? (
        <Row
          label="Objective"
          value={<span className="mono">{formatNumber(obj)}</span>}
        />
      ) : null}
      {n != null ? (
        <Row label="Selected" value={<span className="mono">{n} facilities</span>} />
      ) : null}
      {metric ? (
        <Row label="Distance" value={<span className="mono">{metric}</span>} />
      ) : null}
      {error ? <Row label="Error" value={<span className="text-err">{error}</span>} /> : null}
      {warnings.length > 0 ? (
        <Row
          label="Warnings"
          value={
            <ul className="space-y-0.5 text-warn">
              {warnings.map((w, i) => (
                <li key={i} className="flex items-start gap-1">
                  <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" strokeWidth={1.75} />
                  <span>{w}</span>
                </li>
              ))}
            </ul>
          }
        />
      ) : null}
    </div>
  );
}

function GenericSummary({ summary }: { summary: Summary }) {
  const entries = Object.entries(summary).filter(([, v]) => v != null);
  if (entries.length === 0) return null;
  return (
    <div className="mt-1 space-y-0.5">
      {entries.slice(0, 8).map(([k, v]) => (
        <Row
          key={k}
          label={k}
          value={
            <span className="mono text-text-muted">
              {typeof v === "object" ? JSON.stringify(v) : String(v)}
            </span>
          }
        />
      ))}
    </div>
  );
}

function SummaryRenderer({
  name,
  summary,
}: {
  name: string;
  summary: Summary;
}) {
  if (name === "fetch_city_data") return <FetchCityDataSummary summary={summary} />;
  if (name === "stage_optimization") return <StageOptimizationSummary summary={summary} />;
  if (name === "confirm_optimization") return <ConfirmOptimizationSummary summary={summary} />;
  return <GenericSummary summary={summary} />;
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
  const argLine = prettyArgs(tc.name, tc.args);

  // Pull out unique sources seen in nested activity for an at-a-glance badge.
  const sources = Array.from(
    new Set(
      (tc.activity || [])
        .map((a) => a.source)
        .filter((s): s is string => Boolean(s)),
    ),
  );

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
          {argLine ? (
            <span className="ml-2 text-2xs text-text-faint">· {argLine}</span>
          ) : null}
        </span>
        {done ? (
          failed ? (
            <XCircle className="h-3.5 w-3.5 text-err" strokeWidth={1.75} />
          ) : (
            <CheckCircle2 className="h-3.5 w-3.5 text-ok" strokeWidth={1.75} />
          )
        ) : (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-text-muted" strokeWidth={1.75} />
        )}
        <span className="mono text-2xs text-text-faint tabular-nums w-14 text-right">
          {formatDurationMs(elapsed)}
        </span>
      </button>

      {open ? (
        <div className="hairline-t px-2.5 py-1.5 text-xs text-text-muted">
          {sources.length > 0 ? (
            <div className="mb-1 flex flex-wrap items-center gap-1 text-2xs">
              <span className="text-text-faint">Sources:</span>
              {sources.map((s) => (
                <span
                  key={s}
                  className="rounded border border-border bg-bg px-1.5 py-0.5 mono text-text"
                >
                  {sourceLabel(s)}
                </span>
              ))}
            </div>
          ) : null}

          {tc.activity.length > 0 ? (
            <ul className="mt-0.5 space-y-0.5 border-l border-border pl-2">
              {tc.activity.map((evt, i) => (
                <li key={i} className="flex items-center gap-2 py-0.5 text-2xs">
                  <span className="shrink-0">{statusGlyph(evt.status)}</span>
                  <span className="mono w-32 shrink-0 truncate text-text-muted">
                    {evt.stage}
                  </span>
                  {evt.source ? (
                    <span className="mono shrink-0 text-text-faint">
                      {sourceLabel(evt.source)}
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

          {done && tc.summary ? (
            <div className="mt-2 border-t border-border pt-1.5">
              <SummaryRenderer name={tc.name} summary={tc.summary} />
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
