"use client";

import { useStore } from "@/lib/store";
import { Chip } from "@/components/ui/chip";
import { formatArea, formatNumber } from "@/lib/format";

export function Titlebar() {
  const snapshot = useStore((s) => s.snapshot);
  const network = useStore((s) => s.network);

  const aoi = snapshot?.aoi;
  const networkChip = (() => {
    if (network.status === "ready") {
      const nodes = network.stats?.nodes;
      return (
        <Chip tone="ok">
          road graph · {nodes ? `${formatNumber(nodes)} nodes` : "ready"}
        </Chip>
      );
    }
    if (network.status === "fetching") {
      return <Chip tone="warn">road graph · fetching</Chip>;
    }
    if (network.status === "failed") {
      return <Chip tone="err">road graph · unavailable</Chip>;
    }
    return null;
  })();

  return (
    <header className="hairline-b flex h-11 items-center gap-3 px-3 bg-bg">
      <div className="flex items-center gap-2 select-none">
        <div
          aria-hidden
          className="flex h-5 w-5 items-center justify-center rounded-sm bg-accent text-[10px] font-semibold text-white"
        >
          S
        </div>
        <div className="text-sm font-semibold tracking-tight">SOCA</div>
        <div className="mono text-2xs text-text-faint">
          spatial · optimization · agent
        </div>
      </div>

      <div className="mx-3 h-5 w-px bg-border" />

      <div className="flex items-center gap-2 min-w-0">
        {aoi ? (
          <>
            <Chip tone="accent">{aoi.name}</Chip>
            <span className="mono text-2xs text-text-faint">
              {formatArea(aoi.area_km2)}
            </span>
          </>
        ) : (
          <Chip tone="muted">no area selected</Chip>
        )}
        {networkChip}
      </div>

    </header>
  );
}
