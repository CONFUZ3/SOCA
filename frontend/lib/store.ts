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
  closeOpenToolCalls: (
    turnId: string,
    summary?: Record<string, unknown>,
  ) => void;
  finalizeTurn: (turnId: string, text: string) => void;
  errorTurn: (turnId: string, message: string) => void;
  appendActivity: (evt: ActivityEvent) => void;
  setNetwork: (state: NetworkState) => void;
  setDatasets: (datasets: DatasetSummary[]) => void;
  updateDatasetSubcategories: (name: string, active: string[]) => void;
  updateDatasetSummary: (summary: DatasetSummary) => void;
  resetStore: () => void;
}

const uid = () =>
  `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;

function findTurn(items: ChatItem[], turnId: string): ChatTurn | undefined {
  return items.find(
    (it): it is ChatTurn => it.kind === "turn" && it.id === turnId,
  );
}

/**
 * After an assistant turn ends, any trailing activity-group whose last event
 * is still "try" is stale (the terminal event was lost or never sent). Flip
 * the trailing "try" event's status so the spinner stops.
 */
function sealStaleActivityGroups(
  items: ChatItem[],
  terminalStatus: "ok" | "fail" = "ok",
): ChatItem[] {
  let changed = false;
  const next = items.map((it) => {
    if (it.kind !== "activity-group") return it;
    const lastIdx = it.events.length - 1;
    if (lastIdx < 0 || it.events[lastIdx].status !== "try") return it;
    changed = true;
    const events = it.events.slice();
    events[lastIdx] = { ...events[lastIdx], status: terminalStatus };
    return { ...it, events };
  });
  return changed ? next : items;
}

/**
 * When a tool call finishes, the matching terminal activity events may have
 * already landed elsewhere (or never arrived). Flip any remaining "try"
 * sub-step events to a terminal status so the in-card spinners stop.
 */
function sealToolCallActivity(
  activity: ActivityEvent[],
  terminalStatus: "ok" | "fail",
): ActivityEvent[] {
  let changed = false;
  const next = activity.map((evt) => {
    if (evt.status !== "try") return evt;
    changed = true;
    return { ...evt, status: terminalStatus };
  });
  return changed ? next : activity;
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
    set((state) => {
      // Only seed `items` from the backend's message history on the first
      // snapshot load (or after a reset clears the store). Subsequent
      // refreshes — which fire after every turn to pick up new datasets /
      // network state — must NOT overwrite our richer in-memory items:
      // doing so wipes tool-call `finishedAt` (spinner spins forever) and
      // drops nested activity + summary detail (info "disappears").
      const seedItems = state.items.length === 0;
      const base: Partial<State> = {
        snapshot: s,
        ready: true,
        network: s.network || state.network,
        datasets: s.datasets || [],
      };
      if (!seedItems) return base;
      const items: ChatItem[] = (s.messages || []).map(
        (m: ChatMessage): ChatTurn => ({
          kind: "turn",
          id: uid(),
          role: m.role === "assistant" ? "assistant" : "user",
          content: m.content,
          // Historical tool calls have no live timing/activity to show —
          // mark them finished so the spinner doesn't render against
          // already-completed turns.
          toolCalls: (m.tool_calls || []).map((name) => ({
            name,
            args: {},
            startedAt: 1,
            finishedAt: 1,
            activity: [],
          })),
        }),
      );
      return { ...base, items };
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
        const failed =
          tc.summary &&
          typeof tc.summary === "object" &&
          (("status" in tc.summary && tc.summary.status === "error") ||
            ("staged" in tc.summary && tc.summary.staged === false));
        next[realIdx] = {
          ...next[realIdx],
          summary: tc.summary,
          finishedAt: Date.now(),
          activity: sealToolCallActivity(
            next[realIdx].activity,
            failed ? "fail" : "ok",
          ),
        };
        return { ...t, toolCalls: next };
      }),
    })),

  closeOpenToolCalls: (turnId, summary) =>
    set((st) => ({
      items: mapTurn(st.items, turnId, (t) => ({
        ...t,
        toolCalls: t.toolCalls.map((tc) =>
          tc.finishedAt
            ? tc
            : {
                ...tc,
                summary: summary ?? tc.summary,
                finishedAt: Date.now(),
                activity: sealToolCallActivity(tc.activity, "ok"),
              },
        ),
      })),
    })),

  finalizeTurn: (turnId, text) =>
    set((st) => ({
      items: sealStaleActivityGroups(
        mapTurn(st.items, turnId, (t) => ({
          ...t,
          content: text || t.content,
          pending: false,
        })),
      ),
    })),

  errorTurn: (turnId, message) =>
    set((st) => ({
      items: sealStaleActivityGroups(
        mapTurn(st.items, turnId, (t) => ({
          ...t,
          content: message,
          pending: false,
        })),
        "fail",
      ),
    })),

  appendActivity: (evt) => {
    set((st) => {
      const items = [...st.items];

      // 1. Prefer attaching to an open tool call on the most recent pending
      // assistant turn — even if that turn isn't currently the last item
      // (e.g. an activity-group got pushed after it). Without this, a
      // sub-step event that arrived before tool_call_start would end up in
      // a standalone activity-group, and the matching terminal event would
      // attach to the tool call instead — leaving the group's last event
      // perpetually "try" and its spinner spinning forever.
      for (let i = items.length - 1; i >= 0; i -= 1) {
        const it = items[i];
        if (it.kind !== "turn") continue;
        if (it.role !== "assistant" || !it.pending) break;
        const tcs = it.toolCalls;
        const openIdx = tcs.findIndex((tc) => !tc.finishedAt);
        if (openIdx === -1) break;
        const nextTcs = tcs.slice();
        nextTcs[openIdx] = {
          ...nextTcs[openIdx],
          activity: [...nextTcs[openIdx].activity, evt],
        };
        items[i] = { ...it, toolCalls: nextTcs };
        return { items };
      }

      // 2. Look for a recent activity-group (scanning back a few items, not
      // just the last) — coalesce if close in time, so terminal events
      // catch up with their initiating "try" event even if a tool-call
      // result was interleaved between them.
      const evtMs = evt.timestamp ? evt.timestamp * 1000 : Date.now();
      for (
        let i = items.length - 1;
        i >= Math.max(0, items.length - 4);
        i -= 1
      ) {
        const it = items[i];
        if (it.kind !== "activity-group") continue;
        const lastEvt = it.events[it.events.length - 1];
        const lastMs = lastEvt?.timestamp
          ? lastEvt.timestamp * 1000
          : Date.now();
        if (Math.abs(evtMs - lastMs) < ACTIVITY_GROUP_WINDOW_MS) {
          items[i] = { ...it, events: [...it.events, evt] };
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

  updateDatasetSubcategories: (name, active) =>
    set((st) => ({
      datasets: st.datasets.map((d) =>
        d.name === name ? { ...d, active_subcategories: active } : d,
      ),
    })),

  updateDatasetSummary: (summary) =>
    set((st) => ({
      datasets: st.datasets.map((d) =>
        d.name === summary.name ? { ...d, ...summary } : d,
      ),
    })),

  resetStore: () =>
    set({
      items: [],
      snapshot: null,
      ready: false,
      datasets: [],
      network: { status: null, error: null, stats: null },
    }),
}));
