"use client";

import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { MapState } from "@/types";
import { useStore } from "@/lib/store";

export function useMapState() {
  const snapshot = useStore((s) => s.snapshot);
  const datasets = useStore((s) => s.datasets);
  // The queryKey only needs to react to dataset/filter changes; solution
  // updates arrive via SSE invalidation (use-events-stream + use-chat), which
  // refetches regardless of key. Including snapshot-derived solution_version
  // here used to add timing races with the session-snapshot refetch.
  const dataKey = snapshot?.datasets?.map((d) => d.name).join(",") ?? "";
  const filterKey = datasets
    .map((d) => `${d.name}:${(d.active_subcategories ?? []).join("|")}`)
    .join(",");

  return useQuery({
    queryKey: ["map-state", dataKey, filterKey],
    queryFn: () => apiGet<MapState>("/api/map/state"),
    enabled: Boolean(snapshot?.aoi_confirmed),
    staleTime: 5_000,
    refetchOnWindowFocus: false,
  });
}
