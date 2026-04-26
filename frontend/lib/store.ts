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
  kind: "turn";
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

export interface ActivityGroup {
  kind: "activity-group";
  id: string;
  events: ActivityEvent[];
}

export type ChatItem = ChatTurn | ActivityGroup;

/** Activity events within this many ms get coalesced into the same group. */
const ACTIVITY_GROUP_WINDOW_MS = 4_000;

interface State {
  snapshot: SessionSnapshot | null;
  ready: boolean;
  items: ChatItem[];
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

function findTurn(items: ChatItem[], turnId: string): ChatTurn | undefined {
  return items.find(
    (it): it is ChatTurn => it.kind === "turn" && it.id === turnId,
  );
}

function mapTurn(
  items: ChatItem[],
  turnId: string,
  fn: (t: ChatTurn) => ChatTurn,
): ChatItem[] {
  return items.map((it) =>
    it.kind === "turn" && it.id === turnId ? fn(it) : it,
  );
}

export const useStore = create<State>((set) => ({
  snapshot: null,
  ready: false,
  items: [],
  network: { status: null, error: null, stats: null },
  datasets: [],

  setSnapshot: (s) =>
    set(() => {
      const items: ChatItem[] = (s.messages || []).map(
        (m: ChatMessage): ChatTurn => ({
          kind: "turn",
          id: uid(),
          role: m.role === "assistant" ? "assistant" : "user",
          content: m.content,
          toolCalls: (m.tool_calls || []).map((name) => ({
            name,
            args: {},
            startedAt: 0,
            activity: [],
          })),
        }),
      );
      return {
        snapshot: s,
        ready: true,
        items,
        network: s.network || { status: null },
        datasets: s.datasets || [],
      };
    }),

  appendUserTurn: (text) => {
    const id = uid();
    set((st) => ({
      items: [
        ...st.items,
        {
          kind: "turn",
          id,
          role: "user",
          content: text,
          toolCalls: [],
        },
      ],
    }));
    return id;
  },

  appendAssistantTurn: () => {
    const id = uid();
    set((st) => ({
      items: [
        ...st.items,
        {
          kind: "turn",
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
      items: mapTurn(st.items, turnId, (t) => ({
        ...t,
        content: (t.content || "") + text,
      })),
    })),

  appendToolCallStart: (turnId, tc) =>
    set((st) => ({
      items: mapTurn(st.items, turnId, (t) => ({
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
      })),
    })),

  appendToolCallResult: (turnId, tc) =>
    set((st) => ({
      items: mapTurn(st.items, turnId, (t) => {
        const revIdx = [...t.toolCalls]
          .reverse()
          .findIndex((x) => x.name === tc.name && !x.finishedAt);
        if (revIdx < 0) return t;
        const realIdx = t.toolCalls.length - 1 - revIdx;
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
      items: mapTurn(st.items, turnId, (t) => ({
        ...t,
        content: text || t.content,
        pending: false,
      })),
    })),

  errorTurn: (turnId, message) =>
    set((st) => ({
      items: mapTurn(st.items, turnId, (t) => ({
        ...t,
        content: message,
        pending: false,
      })),
    })),

  appendActivity: (evt) => {
    set((st) => {
      const items = [...st.items];
      const last = items[items.length - 1];

      // 1. Prefer attaching to an open tool call on a pending assistant turn.
      if (last && last.kind === "turn" && last.role === "assistant" && last.pending) {
        const tcs = [...last.toolCalls];
        const openIdx = tcs.length - 1;
        if (openIdx >= 0 && !tcs[openIdx].finishedAt) {
          tcs[openIdx] = {
            ...tcs[openIdx],
            activity: [...tcs[openIdx].activity, evt],
          };
          items[items.length - 1] = { ...last, toolCalls: tcs };
          return { items };
        }
      }

      // 2. Coalesce into a recent activity-group if close in time.
      if (last && last.kind === "activity-group") {
        const lastEvt = last.events[last.events.length - 1];
        const evtMs = evt.timestamp ? evt.timestamp * 1000 : Date.now();
        const lastMs = lastEvt?.timestamp
          ? lastEvt.timestamp * 1000
          : Date.now();
        if (Math.abs(evtMs - lastMs) < ACTIVITY_GROUP_WINDOW_MS) {
          items[items.length - 1] = {
            ...last,
            events: [...last.events, evt],
          };
          return { items };
        }
      }

      // 3. Otherwise push a fresh group.
      items.push({
        kind: "activity-group",
        id: uid(),
        events: [evt],
      });
      return { items };
    });
  },

  setNetwork: (state) => set({ network: state }),

  setDatasets: (datasets) => set({ datasets }),
}));
