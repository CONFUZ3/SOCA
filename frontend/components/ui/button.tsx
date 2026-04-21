"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cn } from "@/lib/cn";

type Variant = "default" | "primary" | "ghost" | "subtle" | "destructive";
type Size = "sm" | "md" | "icon";

const variants: Record<Variant, string> = {
  default:
    "bg-surface text-text border border-border hover:bg-surface-2 hover:border-border-strong",
  primary:
    "bg-accent text-white border border-accent hover:brightness-[1.04] active:brightness-95",
  ghost:
    "bg-transparent text-text hover:bg-surface-2",
  subtle:
    "bg-surface-2 text-text border border-transparent hover:border-border",
  destructive:
    "bg-surface text-err border border-border hover:bg-accent-soft hover:border-err/40",
};

const sizes: Record<Size, string> = {
  sm: "h-7 px-2 text-xs",
  md: "h-8 px-3 text-sm",
  icon: "h-8 w-8 p-0",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", asChild, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        ref={ref}
        className={cn(
          "btn-base",
          "transition-[background-color,border-color,color] duration-[120ms] ease-[cubic-bezier(.2,.7,.2,1)]",
          variants[variant],
          sizes[size],
          className,
        )}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";
