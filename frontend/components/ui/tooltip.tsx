"use client";

import * as React from "react";
import * as TT from "@radix-ui/react-tooltip";
import { cn } from "@/lib/cn";

export function Tooltip({
  content,
  children,
  side = "top",
  align = "center",
  shortcut,
}: {
  content: React.ReactNode;
  children: React.ReactNode;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
  shortcut?: React.ReactNode;
}) {
  return (
    <TT.Root>
      <TT.Trigger asChild>{children}</TT.Trigger>
      <TT.Portal>
        <TT.Content
          side={side}
          align={align}
          sideOffset={6}
          className={cn(
            "z-50 flex items-center gap-2 rounded border border-border bg-surface px-2 py-1 text-2xs text-text",
            "shadow-popover animate-fade-in",
          )}
        >
          <span>{content}</span>
          {shortcut ? (
            <span className="mono text-text-faint">{shortcut}</span>
          ) : null}
        </TT.Content>
      </TT.Portal>
    </TT.Root>
  );
}
