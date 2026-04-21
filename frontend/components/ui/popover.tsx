"use client";

import * as React from "react";
import * as RP from "@radix-ui/react-popover";
import { cn } from "@/lib/cn";

export const Popover = RP.Root;
export const PopoverTrigger = RP.Trigger;

export function PopoverContent({
  className,
  sideOffset = 6,
  ...props
}: React.ComponentProps<typeof RP.Content>) {
  return (
    <RP.Portal>
      <RP.Content
        sideOffset={sideOffset}
        className={cn(
          "z-50 min-w-56 rounded-md border border-border bg-surface p-1 text-sm",
          "shadow-popover animate-fade-in",
          className,
        )}
        {...props}
      />
    </RP.Portal>
  );
}
