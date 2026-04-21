import * as React from "react";
import { cn } from "@/lib/cn";

type Tone = "neutral" | "accent" | "ok" | "warn" | "err" | "muted";

const tones: Record<Tone, string> = {
  neutral: "bg-surface text-text-muted border-border",
  accent: "bg-accent-soft text-accent border-accent/20",
  ok: "bg-surface text-ok border-ok/25",
  warn: "bg-surface text-warn border-warn/25",
  err: "bg-surface text-err border-err/25",
  muted: "bg-surface-2 text-text-faint border-transparent",
};

export function Chip({
  children,
  tone = "neutral",
  className,
  icon,
}: {
  children: React.ReactNode;
  tone?: Tone;
  className?: string;
  icon?: React.ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm border px-1.5 py-0.5 text-2xs font-medium",
        "whitespace-nowrap",
        tones[tone],
        className,
      )}
    >
      {icon ? <span className="inline-flex">{icon}</span> : null}
      {children}
    </span>
  );
}
