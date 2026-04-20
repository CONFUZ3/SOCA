"use client";

import { useEffect } from "react";
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

  const q = useQuery({
    queryKey: ["session"],
    queryFn: ensureSession,
    staleTime: 30_000,
  });

  useEffect(() => {
    if (q.data) setSnapshot(q.data);
  }, [q.data, setSnapshot]);

  return {
    ...q,
    refresh: () => qc.invalidateQueries({ queryKey: ["session"] }),
  };
}
