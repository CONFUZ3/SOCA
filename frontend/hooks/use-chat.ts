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
  | { event: "map_update"; data: { name?: string } }
  | { event: "final"; data: { text: string; tool_calls: string[] } }
  | { event: "error"; data: { message: string } };

export function useChat(onTurnFinished?: () => void, onMapUpdate?: () => void) {
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const {
    appendUserTurn,
    appendAssistantTurn,
    appendAssistantToken,
    appendToolCallStart,
    appendToolCallResult,
    closeOpenToolCalls,
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

      let receivedFinal = false;
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
            case "map_update":
              onMapUpdate?.();
              break;
            case "final":
              receivedFinal = true;
              closeOpenToolCalls(assistantId, { status: "completed" });
              finalizeTurn(assistantId, f.data.text || "");
              break;
            case "error":
              receivedFinal = true;
              closeOpenToolCalls(assistantId, {
                error: f.data.message || "Something went wrong.",
              });
              errorTurn(assistantId, f.data.message || "Something went wrong.");
              break;
            default:
              break;
          }
        }
        // Stream ended without a final event (server detected disconnect and
        // cancelled the pump mid-turn). Close any open tool calls so the UI
        // doesn't spin forever; the map will update via the events stream.
        if (!receivedFinal) {
          closeOpenToolCalls(assistantId, { status: "completed" });
          finalizeTurn(assistantId, "");
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        closeOpenToolCalls(assistantId, { error: `Stream interrupted: ${msg}` });
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
      closeOpenToolCalls,
      finalizeTurn,
      errorTurn,
      onTurnFinished,
      onMapUpdate,
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
