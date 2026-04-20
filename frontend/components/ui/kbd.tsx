import * as React from "react";
import { cn } from "@/lib/cn";

export function Kbd({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <kbd
      className={cn(
        "inline-flex h-5 min-w-5 items-center justify-center rounded-sm border border-border bg-surface px-1 font-mono text-2xs text-text-muted",
        className,
      )}
    >
      {children}
    </kbd>
  );
}
