"use client";

import { useEffect, useState } from "react";
import { Loader2, Network, CheckCircle2, RefreshCw, XCircle } from "lucide-react";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/cn";
import { formatNumber } from "@/lib/format";
import { apiPost, ApiError } from "@/lib/api";

/**
 * Thin status strip at the top of the chat panel that surfaces the
 * road-network fetch lifecycle. Hidden when idle; auto-dismisses the
 * "ready" state after a few seconds so it doesn't clutter the UI.
 */
export function NetworkStatusStrip() {
  const network = useStore((s) => s.network);
  const setNetwork = useStore((s) => s.setNetwork);
  const [dismissedReady, setDismissedReady] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [retryError, setRetryError] = useState<string | null>(null);

  const handleRetry = async () => {
    setRetrying(true);
    setRetryError(null);
    try {
      await apiPost("/api/network/refresh");
      // The SSE stream will push the new "fetching" / "ready" status, but
      // flip to "fetching" locally for instant feedback.
      setNetwork({ status: "fetching", error: null, stats: null });
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? typeof e.detail === "string"
            ? e.detail
            : e.message
          : e instanceof Error
            ? e.message
            : "Retry failed";
      setRetryError(msg);
    } finally {
      setRetrying(false);
    }
  };

  useEffect(() => {
    if (network.status === "ready") {
      setDismissedReady(false);
      const t = setTimeout(() => setDismissedReady(true), 5000);
      return () => clearTimeout(t);
    }
    setDismissedReady(false);
  }, [network.status, network.stats?.nodes, network.stats?.edges]);

  if (!network.status) return null;
  if (network.status === "ready" && dismissedReady) return null;

  if (network.status === "fetching") {
    return (
      <div
        className={cn(
          "flex items-center gap-2 border-b border-border bg-surface-2 px-3 py-1.5 text-2xs text-text-muted",
        )}
      >
        <Loader2 className="h-3 w-3 animate-spin" strokeWidth={1.75} />
        <Network className="h-3 w-3 text-text-faint" strokeWidth={1.5} />
        <span className="flex-1">
          Downloading road network from OpenStreetMap…
        </span>
      </div>
    );
  }

  if (network.status === "ready") {
    const nodes = network.stats?.nodes;
    const edges = network.stats?.edges;
    return (
      <div className="flex items-center gap-2 border-b border-border bg-surface-2 px-3 py-1.5 text-2xs text-text-muted">
        <CheckCircle2 className="h-3 w-3 text-ok" strokeWidth={1.75} />
        <Network className="h-3 w-3 text-text-faint" strokeWidth={1.5} />
        <span className="flex-1">
          Road network ready
          {nodes != null && edges != null ? (
            <span className="mono ml-1.5 text-text-faint">
              · {formatNumber(nodes)} nodes · {formatNumber(edges)} edges
            </span>
          ) : null}
        </span>
      </div>
    );
  }

  // failed
  return (
    <div className="flex items-center gap-2 border-b border-err/40 bg-err/10 px-3 py-1.5 text-2xs text-err">
      <XCircle className="h-3 w-3" strokeWidth={1.75} />
      <Network className="h-3 w-3" strokeWidth={1.5} />
      <span className="flex-1 truncate">
        Road network unavailable
        {network.error ? ` · ${network.error}` : ""}
        {retryError ? ` · retry: ${retryError}` : ""}. Solver falls back to
        geodesic.
      </span>
      <button
        type="button"
        onClick={() => void handleRetry()}
        disabled={retrying}
        className={cn(
          "flex items-center gap-1 rounded border border-err/40 bg-err/10 px-1.5 py-0.5 mono text-err",
          "hover:bg-err/20 disabled:opacity-50",
        )}
      >
        {retrying ? (
          <Loader2 className="h-3 w-3 animate-spin" strokeWidth={1.75} />
        ) : (
          <RefreshCw className="h-3 w-3" strokeWidth={1.75} />
        )}
        Retry
      </button>
    </div>
  );
}
