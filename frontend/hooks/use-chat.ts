"use client";

import { useCallback, useRef, useState } from "react";
import { streamPostSse } from "@/lib/sse";
import { useStore } from "@/lib/store";
import type { ToolCallResult, ToolCallStart } from "@/types";

type Frame =
  | { event: "start"; data: { ok: boolean } }
  | { event: "token"; data: { text: string } }
  | { event: "tool_call_start"; data: ToolCallStart }
  | { event: "tool_call_result"; data: ToolCallResult }
  | { event: "final"; data: { text: string; tool_calls: string[] } }
  | { event: "error"; data: { message: string } };

export function useChat(onTurnFinished?: () => void) {
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const {
    appendUserTurn,
    appendAssistantTurn,
    appendAssistantToken,
    appendToolCallStart,
    appendToolCallResult,
    finalizeTurn,
    errorTurn,
  } = useStore.getState();

  const send = useCallback(
    async (message: string) => {
      if (!message.trim()) return;
      if (busy) return;
      setBusy(true);

      appendUserTurn(message);
      const assistantId = appendAssistantTurn();

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      try {
        for await (const frame of streamPostSse<unknown>(
          "/api/chat/stream",
          { message },
          ctrl.signal,
        )) {
          const f = frame as Frame;
          switch (f.event) {
            case "token":
              appendAssistantToken(assistantId, f.data.text || "");
              break;
            case "tool_call_start":
              appendToolCallStart(assistantId, f.data);
              break;
            case "tool_call_result":
              appendToolCallResult(assistantId, f.data);
              break;
            case "final":
              finalizeTurn(assistantId, f.data.text || "");
              break;
            case "error":
              errorTurn(assistantId, f.data.message || "Something went wrong.");
              break;
            default:
              break;
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        errorTurn(assistantId, `Stream interrupted: ${msg}`);
      } finally {
        setBusy(false);
        abortRef.current = null;
        onTurnFinished?.();
      }
    },
    [
      busy,
      appendUserTurn,
      appendAssistantTurn,
      appendAssistantToken,
      appendToolCallStart,
      appendToolCallResult,
      finalizeTurn,
      errorTurn,
      onTurnFinished,
    ],
  );

  const cancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
      setBusy(false);
    }
  }, []);

  return { send, cancel, busy };
}
