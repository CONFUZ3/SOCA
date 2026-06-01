"use client";

import { useCallback, useRef, useState } from "react";
import { apiGet } from "@/lib/api";
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
        // Stream ended without a final event (server detected a disconnect
        // mid-turn). The backend worker still finishes the turn and persists
        // the assistant message, so recover it from chat history rather than
        // leaving the turn blank; also refresh the map for any new layers.
        if (!receivedFinal) {
          closeOpenToolCalls(assistantId, { status: "completed" });
          // The worker may still be generating the narration when the stream
          // drops, so poll a few times for an assistant message that lands
          // AFTER this turn's user message (older assistant messages belong
          // to previous turns and must not be shown here).
          let recovered = "";
          for (let attempt = 0; attempt < 4 && !recovered; attempt++) {
            if (attempt > 0) await new Promise((r) => setTimeout(r, 1500));
            try {
              const { messages } = await apiGet<{
                messages: { role: string; content: string }[];
              }>("/api/chat/history");
              const msgs = messages || [];
              const lastUserIdx = msgs.map((m) => m.role).lastIndexOf("user");
              recovered =
                msgs
                  .slice(lastUserIdx + 1)
                  .find((m) => m.role === "assistant")?.content || "";
            } catch {
              break;
            }
          }
          finalizeTurn(assistantId, recovered);
          onMapUpdate?.();
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
