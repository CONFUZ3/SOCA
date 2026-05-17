"use client";

import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { MapState } from "@/types";
import { useStore } from "@/lib/store";

export function useMapState() {
  const snapshot = useStore((s) => s.snapshot);
  const datasets = useStore((s) => s.datasets);
  // Refresh when datasets change, solution changes, AOI changes, or subcategory filters change.
  const dataKey = snapshot?.datasets?.map((d) => d.name).join(",") ?? "";
  const hasSolution = snapshot?.has_solution ?? false;
  const solutionStatus = snapshot?.solution_status ?? "";
  const solutionVersion = snapshot?.solution_version ?? 0;
  const filterKey = datasets
    .map((d) => `${d.name}:${(d.active_subcategories ?? []).join("|")}`)
    .join(",");

  return useQuery({
    queryKey: ["map-state", dataKey, hasSolution, solutionStatus, solutionVersion, filterKey],
    queryFn: () => apiGet<MapState>("/api/map/state"),
    enabled: Boolean(snapshot?.aoi_confirmed),
    staleTime: 5_000,
    refetchOnWindowFocus: false,
  });
}
