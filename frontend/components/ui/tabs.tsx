"use client";

import * as React from "react";
import * as RT from "@radix-ui/react-tabs";
import { cn } from "@/lib/cn";

export const Tabs = RT.Root;

export function TabsList({
  className,
  ...props
}: React.ComponentProps<typeof RT.List>) {
  return (
    <RT.List
      className={cn(
        "inline-flex h-8 items-center gap-1 rounded-sm border border-border bg-surface-2 p-0.5",
        className,
      )}
      {...props}
    />
  );
}

export function TabsTrigger({
  className,
  ...props
}: React.ComponentProps<typeof RT.Trigger>) {
  return (
    <RT.Trigger
      className={cn(
        "inline-flex h-7 items-center rounded-[3px] px-2 text-xs font-medium text-text-muted",
        "transition-colors duration-[120ms]",
        "hover:text-text",
        "data-[state=active]:bg-surface data-[state=active]:text-text data-[state=active]:shadow-hairline",
        className,
      )}
      {...props}
    />
  );
}

export const TabsContent = RT.Content;
