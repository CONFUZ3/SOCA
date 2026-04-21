"use client";

import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { MapState } from "@/types";
import { useStore } from "@/lib/store";

export function useMapState() {
  const snapshot = useStore((s) => s.snapshot);
  // Refresh when datasets change, solution changes, or AOI changes.
  const dataKey = snapshot?.datasets?.map((d) => d.name).join(",") ?? "";
  const hasSolution = snapshot?.has_solution ?? false;

  return useQuery({
    queryKey: ["map-state", dataKey, hasSolution],
    queryFn: () => apiGet<MapState>("/api/map/state"),
    enabled: Boolean(snapshot?.aoi_confirmed),
    staleTime: 5_000,
    refetchOnWindowFocus: false,
  });
}
