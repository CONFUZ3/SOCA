"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatTurn } from "@/lib/store";
import { cn } from "@/lib/cn";
import { ToolCallCard } from "./tool-call-card";

export function Message({ turn }: { turn: ChatTurn }) {
  if (turn.role === "user") {
    return (
      <div className="flex justify-end py-1.5">
        <div className="max-w-[80%] rounded bg-surface-2 px-2.5 py-1.5 text-sm text-text">
          <div className="whitespace-pre-wrap leading-relaxed">{turn.content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="py-2">
      <div className="mb-1 text-2xs font-medium uppercase tracking-wider text-text-faint">
        SOCA
      </div>

      {turn.toolCalls.length > 0 ? (
        <div className="mb-1">
          {turn.toolCalls.map((tc, i) => (
            <ToolCallCard tc={tc} key={`${tc.name}-${i}-${tc.startedAt}`} />
          ))}
        </div>
      ) : null}

      <div
        className={cn(
          "prose-soca max-w-none text-sm leading-relaxed text-text",
          turn.pending && !turn.content ? "min-h-5" : "",
        )}
      >
        {turn.content ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {turn.content}
          </ReactMarkdown>
        ) : turn.pending ? (
          <span className="inline-block h-4 w-2 translate-y-0.5 animate-blink-caret bg-text-muted align-middle" />
        ) : null}
        {turn.pending && turn.content ? (
          <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-blink-caret bg-text-muted align-middle" />
        ) : null}
      </div>
    </div>
  );
}
