/**
 * Client-side store for session snapshot + live streams.
 */

import { create } from "zustand";
import type {
  ActivityEvent,
  ChatMessage,
  DatasetSummary,
  NetworkState,
  SessionSnapshot,
  ToolCallResult,
  ToolCallStart,
} from "@/types";

export interface ChatTurn {
  /** Stable id for React keying. */
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls: ChatTurnToolCall[];
  /** True while the assistant turn is still streaming. */
  pending?: boolean;
}

export interface ChatTurnToolCall {
  name: string;
  args: Record<string, unknown>;
  summary?: Record<string, unknown>;
  startedAt: number;
  finishedAt?: number;
  activity: ActivityEvent[];
}

interface State {
  snapshot: SessionSnapshot | null;
  ready: boolean;
  turns: ChatTurn[];
  activity: ActivityEvent[];
  network: NetworkState;
  datasets: DatasetSummary[];

  setSnapshot: (s: SessionSnapshot) => void;
  appendUserTurn: (text: string) => string;
  appendAssistantTurn: () => string;
  appendAssistantToken: (turnId: string, text: string) => void;
  appendToolCallStart: (turnId: string, tc: ToolCallStart) => void;
  appendToolCallResult: (turnId: string, tc: ToolCallResult) => void;
  finalizeTurn: (turnId: string, text: string) => void;
  errorTurn: (turnId: string, message: string) => void;
  appendActivity: (evt: ActivityEvent) => void;
  setNetwork: (state: NetworkState) => void;
  setDatasets: (datasets: DatasetSummary[]) => void;
}

const uid = () =>
  `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;

export const useStore = create<State>((set, get) => ({
  snapshot: null,
  ready: false,
  turns: [],
  activity: [],
  network: { status: null, error: null, stats: null },
  datasets: [],

  setSnapshot: (s) =>
    set(() => {
      const turns: ChatTurn[] = (s.messages || []).map((m: ChatMessage) => ({
        id: uid(),
        role: m.role === "assistant" ? "assistant" : "user",
        content: m.content,
        toolCalls: (m.tool_calls || []).map((name) => ({
          name,
          args: {},
          startedAt: 0,
          activity: [],
        })),
      }));
      return {
        snapshot: s,
        ready: true,
        turns,
        network: s.network || { status: null },
        datasets: s.datasets || [],
      };
    }),

  appendUserTurn: (text) => {
    const id = uid();
    set((st) => ({
      turns: [
        ...st.turns,
        { id, role: "user", content: text, toolCalls: [] },
      ],
    }));
    return id;
  },

  appendAssistantTurn: () => {
    const id = uid();
    set((st) => ({
      turns: [
        ...st.turns,
        {
          id,
          role: "assistant",
          content: "",
          toolCalls: [],
          pending: true,
        },
      ],
    }));
    return id;
  },

  appendAssistantToken: (turnId, text) =>
    set((st) => ({
      turns: st.turns.map((t) =>
        t.id === turnId
          ? { ...t, content: (t.content || "") + text }
          : t,
      ),
    })),

  appendToolCallStart: (turnId, tc) =>
    set((st) => ({
      turns: st.turns.map((t) =>
        t.id === turnId
          ? {
              ...t,
              toolCalls: [
                ...t.toolCalls,
                {
                  name: tc.name,
                  args: tc.args || {},
                  startedAt: Date.now(),
                  activity: [],
                },
              ],
            }
          : t,
      ),
    })),

  appendToolCallResult: (turnId, tc) =>
    set((st) => ({
      turns: st.turns.map((t) => {
        if (t.id !== turnId) return t;
        const idx = [...t.toolCalls].reverse().findIndex(
          (x) => x.name === tc.name && !x.finishedAt,
        );
        if (idx < 0) return t;
        const realIdx = t.toolCalls.length - 1 - idx;
        const next = [...t.toolCalls];
        next[realIdx] = {
          ...next[realIdx],
          summary: tc.summary,
          finishedAt: Date.now(),
        };
        return { ...t, toolCalls: next };
      }),
    })),

  finalizeTurn: (turnId, text) =>
    set((st) => ({
      turns: st.turns.map((t) =>
        t.id === turnId
          ? { ...t, content: text || t.content, pending: false }
          : t,
      ),
    })),

  errorTurn: (turnId, message) =>
    set((st) => ({
      turns: st.turns.map((t) =>
        t.id === turnId
          ? { ...t, content: message, pending: false }
          : t,
      ),
    })),

  appendActivity: (evt) => {
    set((st) => {
      const next = [...st.activity, evt];
      if (next.length > 100) next.splice(0, next.length - 100);
      // Attach to the currently-pending tool call, if any, so the inline
      // activity list stays synchronised with its parent.
      const turns = [...st.turns];
      const last = turns[turns.length - 1];
      if (last && last.role === "assistant" && last.pending) {
        const tcs = [...last.toolCalls];
        const openIdx = tcs.length - 1;
        if (openIdx >= 0 && !tcs[openIdx].finishedAt) {
          tcs[openIdx] = {
            ...tcs[openIdx],
            activity: [...tcs[openIdx].activity, evt],
          };
          turns[turns.length - 1] = { ...last, toolCalls: tcs };
        }
      }
      return { activity: next, turns };
    });
  },

  setNetwork: (state) => set({ network: state }),

  setDatasets: (datasets) => set({ datasets }),
}));
