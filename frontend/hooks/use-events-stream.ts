"use client";

import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { subscribeGetSse } from "@/lib/sse";
import { useStore } from "@/lib/store";
import type { ActivityEvent, NetworkSseEvent } from "@/types";

/**
 * Subscribe to /api/events/stream once the session is ready and mirror
 * activity + network events into the Zustand store.
 *
 * The EventSource auto-reconnects on transient failures.
 */
export function useEventsStream(enabled: boolean) {
  const ref = useRef<(() => void) | null>(null);
  const appendActivity = useStore((s) => s.appendActivity);
  const setNetwork = useStore((s) => s.setNetwork);
  const qc = useQueryClient();

  useEffect(() => {
    if (!enabled) return;
    if (ref.current) ref.current();
    const unsub = subscribeGetSse("/api/events/stream", (evt) => {
      if (evt.event === "activity") {
        appendActivity(evt.data as ActivityEvent);
      } else if (evt.event === "network") {
        const d = evt.data as NetworkSseEvent;
        setNetwork({
          status:
            d.status === "fetching" || d.status === "ready" || d.status === "failed"
              ? d.status
              : null,
          error: d.error ?? null,
          stats: d.stats ?? null,
        });
      } else if (evt.event === "solution_ready") {
        qc.invalidateQueries({ queryKey: ["session"] });
        // Force a network refetch directly so the new solution appears the
        // moment confirm_optimization writes it, regardless of staleTime.
        qc.refetchQueries({ queryKey: ["map-state"] });
      }
    });
    ref.current = unsub;
    return () => {
      try {
        unsub();
      } catch {
        /* ignore */
      }
      ref.current = null;
    };
  }, [enabled, appendActivity, setNetwork, qc]);
}
