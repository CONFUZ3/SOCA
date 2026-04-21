"use client";

import * as React from "react";
import { cn } from "@/lib/cn";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, type = "text", ...props }, ref) => {
  return (
    <input
      ref={ref}
      type={type}
      className={cn(
        "h-8 w-full rounded border border-border bg-surface px-2 text-sm text-text",
        "placeholder:text-text-faint",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:border-border-strong",
        "disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
});
Input.displayName = "Input";
