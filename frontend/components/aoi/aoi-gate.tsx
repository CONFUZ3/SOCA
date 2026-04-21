"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { apiGet, apiPost } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Chip } from "@/components/ui/chip";
import { useStore } from "@/lib/store";
import { useSession } from "@/hooks/use-session";
import { ArrowRight, Loader2, MapPin, Search } from "lucide-react";
import { cn } from "@/lib/cn";
import { formatArea } from "@/lib/format";
import type { GeocodeCandidate } from "@/types";

interface ResolvedAoi {
  name: string;
  display_name: string;
  source: string;
  area_km2: number;
  geojson: unknown;
}

export function AoiGate() {
  const [q, setQ] = useState("");
  const [suggestions, setSuggestions] = useState<GeocodeCandidate[]>([]);
  const [loadingSuggest, setLoadingSuggest] = useState(false);
  const [selected, setSelected] = useState<GeocodeCandidate | null>(null);
  const [resolved, setResolved] = useState<ResolvedAoi | null>(null);
  const [resolving, setResolving] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { refresh } = useSession();

  const debounced = useDebouncedValue(q, 220);

  useEffect(() => {
    if (!debounced || debounced.trim().length < 3) {
      setSuggestions([]);
      return;
    }
    let cancelled = false;
    setLoadingSuggest(true);
    apiGet<{ candidates: GeocodeCandidate[] }>(
      `/api/aoi/suggest?q=${encodeURIComponent(debounced.trim())}&limit=6`,
    )
      .then((r) => {
        if (!cancelled) setSuggestions(r.candidates || []);
      })
      .catch(() => {
        if (!cancelled) setSuggestions([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingSuggest(false);
      });
    return () => {
      cancelled = true;
    };
  }, [debounced]);

  const onPick = async (c: GeocodeCandidate) => {
    setSelected(c);
    setResolved(null);
    setError(null);
    setResolving(true);
    try {
      const r = await apiPost<ResolvedAoi>("/api/aoi/resolve", {
        candidate: c,
      });
      setResolved(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to resolve boundary.");
    } finally {
      setResolving(false);
    }
  };

  const onConfirm = async () => {
    if (!resolved) return;
    setConfirming(true);
    setError(null);
    try {
      await apiPost("/api/aoi/confirm", {
        name: resolved.name,
        source: resolved.source,
        geojson: resolved.geojson,
      });
      refresh();
    } catch (e: unknown) {
      setError(
        e instanceof Error ? e.message : "Failed to confirm area of interest.",
      );
    } finally {
      setConfirming(false);
    }
  };

  return (
    <div className="flex h-full w-full items-center justify-center bg-bg px-6 py-10">
      <div className="mx-auto w-full max-w-xl">
        <div className="mb-6">
          <div className="text-2xs font-medium uppercase tracking-wider text-text-faint">
            Step 1 of 2
          </div>
          <h1 className="heading-panel mt-1">Pick an area of interest</h1>
          <p className="mt-1.5 text-sm text-text-muted">
            Type a city, district, or country. SOCA fetches the real boundary
            from OpenStreetMap, not a bounding box.
          </p>
        </div>

        <div className="relative">
          <Search
            className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-faint"
            strokeWidth={1.5}
          />
          <Input
            className="h-10 pl-7 text-md"
            placeholder="e.g. Lima, Peru"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            autoFocus
          />
          {loadingSuggest ? (
            <Loader2
              className="absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 animate-spin text-text-faint"
              strokeWidth={1.75}
            />
          ) : null}
        </div>

        {suggestions.length > 0 && !resolved ? (
          <ul className="mt-2 overflow-hidden rounded-md border border-border bg-surface">
            {suggestions.map((c, i) => (
              <li key={`${c.display_name}-${i}`}>
                <button
                  type="button"
                  onClick={() => onPick(c)}
                  className={cn(
                    "flex w-full items-center gap-2 border-b border-border px-2.5 py-2 text-left text-sm last:border-b-0",
                    "transition-colors hover:bg-surface-2",
                    selected?.display_name === c.display_name && "bg-surface-2",
                  )}
                >
                  <MapPin
                    className="h-3.5 w-3.5 shrink-0 text-text-faint"
                    strokeWidth={1.5}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2">
                      <span className="truncate text-text">{c.short_name}</span>
                      {c.kind ? (
                        <span className="mono text-2xs text-text-faint">
                          {c.kind}
                        </span>
                      ) : null}
                    </div>
                    {c.context ? (
                      <div className="truncate text-xs text-text-muted">
                        {c.context}
                      </div>
                    ) : null}
                  </div>
                  {c.has_relation ? (
                    <Chip tone="ok">real boundary</Chip>
                  ) : (
                    <Chip tone="muted">bbox</Chip>
                  )}
                </button>
              </li>
            ))}
          </ul>
        ) : null}

        {resolving ? (
          <div className="mt-3 flex items-center gap-2 text-xs text-text-muted">
            <Loader2 className="h-3 w-3 animate-spin" strokeWidth={1.75} />
            Fetching boundary polygon from OpenStreetMap…
          </div>
        ) : null}

        {resolved ? (
          <div className="mt-3 rounded-md border border-border bg-surface p-3">
            <div className="flex items-start gap-2">
              <MapPin
                className="mt-0.5 h-4 w-4 shrink-0 text-accent"
                strokeWidth={1.75}
              />
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-text">
                  {resolved.name}
                </div>
                <div className="truncate text-xs text-text-muted">
                  {resolved.display_name}
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <Chip tone="accent">{formatArea(resolved.area_km2)}</Chip>
                  <Chip tone="muted">source · {resolved.source}</Chip>
                </div>
              </div>
            </div>
            <div className="mt-3 flex items-center justify-end gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setResolved(null);
                  setSelected(null);
                }}
              >
                Back
              </Button>
              <Button
                variant="primary"
                size="md"
                onClick={onConfirm}
                disabled={confirming}
              >
                Confirm area
                <ArrowRight className="h-3.5 w-3.5" strokeWidth={2} />
              </Button>
            </div>
          </div>
        ) : null}

        {error ? (
          <div className="mt-3 rounded border border-err/30 bg-accent-soft px-2.5 py-1.5 text-xs text-err">
            {error}
          </div>
        ) : null}

        <div className="mt-6 flex flex-wrap items-center gap-1.5 text-2xs text-text-faint">
          <span className="mono">Try</span>
          {["Lima, Peru", "Brooklyn, New York", "Nairobi, Kenya", "Mirpur, Dhaka"].map(
            (ex) => (
              <button
                key={ex}
                type="button"
                className="rounded-sm border border-border bg-surface px-1.5 py-0.5 text-text-muted hover:bg-surface-2"
                onClick={() => setQ(ex)}
              >
                {ex}
              </button>
            ),
          )}
        </div>
      </div>
    </div>
  );
}

function useDebouncedValue<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  const ref = useRef<number | null>(null);
  useEffect(() => {
    if (ref.current) window.clearTimeout(ref.current);
    ref.current = window.setTimeout(() => setV(value), ms);
    return () => {
      if (ref.current) window.clearTimeout(ref.current);
    };
  }, [value, ms]);
  return v;
}
