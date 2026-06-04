/**
 * Number/date/distance formatters. Shared so the UI reads consistently.
 */

const nf = new Intl.NumberFormat("en-US");
const nfCompact = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

export function formatNumber(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 10_000) return nfCompact.format(n);
  return nf.format(Math.round(n * 100) / 100);
}

export function formatArea(km2: number | null | undefined): string {
  if (km2 === null || km2 === undefined || Number.isNaN(km2)) return "—";
  return `${nf.format(Math.round(km2 * 10) / 10)} km²`;
}

export function formatDurationMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
  const mins = Math.floor(ms / 60_000);
  const secs = Math.round((ms % 60_000) / 1000);
  return `${mins}m ${secs}s`;
}

export function formatDistanceMeters(
  m: number | null | undefined,
  preferredUnit: "m" | "km" | "mi" | "ft" | "yd" = "m",
): string {
  if (m === null || m === undefined || Number.isNaN(m)) return "—";
  if (preferredUnit === "km") return `${(m / 1000).toFixed(2)} km`;
  if (preferredUnit === "mi") return `${(m / 1609.344).toFixed(2)} mi`;
  if (preferredUnit === "ft") return `${(m / 0.3048).toFixed(1)} ft`;
  if (preferredUnit === "yd") return `${(m / 0.9144).toFixed(1)} yd`;
  if (m >= 1000) return `${(m / 1000).toFixed(2)} km`;
  return `${Math.round(m)} m`;
}

export function shortNumber(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return nfCompact.format(n);
}
