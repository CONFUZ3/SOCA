"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowUp, Square } from "lucide-react";
import { cn } from "@/lib/cn";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { Tooltip } from "@/components/ui/tooltip";

export function Composer({
  onSend,
  onCancel,
  busy,
  placeholder = "Describe what to optimize (e.g. place 5 clinics to maximize 2 km coverage)",
}: {
  onSend: (text: string) => void;
  onCancel?: () => void;
  busy: boolean;
  placeholder?: string;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    // Cmd/Ctrl-/ focus shortcut.
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "/") {
        e.preventDefault();
        ref.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const autosize = () => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  const submit = () => {
    if (!value.trim() || busy) return;
    onSend(value.trim());
    setValue("");
    requestAnimationFrame(autosize);
  };

  return (
    <div
      className={cn(
        "relative flex items-end gap-2 rounded-md border border-border bg-surface p-1.5",
        "focus-within:border-border-strong",
      )}
    >
      <textarea
        ref={ref}
        rows={1}
        value={value}
        placeholder={placeholder}
        onChange={(e) => {
          setValue(e.target.value);
          autosize();
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        className={cn(
          "block flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-text outline-none",
          "placeholder:text-text-faint",
        )}
      />
      {busy ? (
        <Tooltip content="Stop generation">
          <Button
            size="icon"
            variant="subtle"
            onClick={onCancel}
            aria-label="Stop"
          >
            <Square className="h-3.5 w-3.5" strokeWidth={1.75} />
          </Button>
        </Tooltip>
      ) : (
        <Tooltip
          content={
            <>
              Send <Kbd>↵</Kbd>
            </>
          }
        >
          <Button
            size="icon"
            variant="primary"
            onClick={submit}
            disabled={!value.trim()}
            aria-label="Send"
          >
            <ArrowUp className="h-4 w-4" strokeWidth={2} />
          </Button>
        </Tooltip>
      )}
    </div>
  );
}
