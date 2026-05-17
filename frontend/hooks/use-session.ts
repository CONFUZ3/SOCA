"use client";

import { useCallback, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiGet, apiPost } from "@/lib/api";
import type { SessionSnapshot } from "@/types";
import { useStore } from "@/lib/store";

async function ensureSession(): Promise<SessionSnapshot> {
  try {
    return await apiGet<SessionSnapshot>("/api/session");
  } catch {
    return await apiPost<SessionSnapshot>("/api/session");
  }
}

export function useSession() {
  const qc = useQueryClient();
  const setSnapshot = useStore((s) => s.setSnapshot);
  const resetStore = useStore((s) => s.resetStore);

  const q = useQuery({
    queryKey: ["session"],
    queryFn: ensureSession,
    staleTime: 30_000,
  });

  useEffect(() => {
    if (q.data) setSnapshot(q.data);
  }, [q.data, setSnapshot]);

  const resetSession = useCallback(async () => {
    await fetch("/api/session", { method: "DELETE", credentials: "include" });
    resetStore();
    qc.invalidateQueries({ queryKey: ["session"] });
    qc.invalidateQueries({ queryKey: ["map-state"] });
  }, [qc, resetStore]);

  return {
    ...q,
    refresh: () => {
      qc.invalidateQueries({ queryKey: ["session"] });
      qc.invalidateQueries({ queryKey: ["map-state"] });
    },
    resetSession,
  };
}
