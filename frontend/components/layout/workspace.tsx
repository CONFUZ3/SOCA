"use client";

import dynamic from "next/dynamic";
import { useSession } from "@/hooks/use-session";
import { useEventsStream } from "@/hooks/use-events-stream";
import { useStore } from "@/lib/store";
import { useMapState } from "@/hooks/use-map-state";
import { AoiGate } from "@/components/aoi/aoi-gate";
import { ChatPanel } from "@/components/chat/chat-panel";
import { Sidebar } from "@/components/layout/sidebar";
import { Titlebar } from "@/components/layout/titlebar";
import { Loader2, Map } from "lucide-react";

// maplibre-gl uses browser APIs — skip SSR entirely
const MapView = dynamic(
  () => import("@/components/map/map-view").then((m) => m.MapView),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full w-full items-center justify-center bg-mono-surface">
        <Loader2 className="h-4 w-4 animate-spin text-text-faint" strokeWidth={1.5} />
      </div>
    ),
  },
);

function MapSection() {
  const { data: mapState, isLoading, isError } = useMapState();

  if (isLoading) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-mono-surface">
        <Loader2 className="h-4 w-4 animate-spin text-text-faint" strokeWidth={1.5} />
      </div>
    );
  }

  if (isError || !mapState) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-mono-surface">
        <Map className="h-5 w-5 text-text-faint" strokeWidth={1.25} />
        <p className="text-xs text-text-faint">Map unavailable</p>
      </div>
    );
  }

  return <MapView state={mapState} />;
}

export function Workspace() {
  const { isLoading, isError } = useSession();
  const ready = useStore((s) => s.ready);
  const snapshot = useStore((s) => s.snapshot);

  useEventsStream(ready);

  if (isLoading || !ready) {
    return (
      <div className="flex h-dvh w-dvw items-center justify-center bg-bg">
        <div className="flex items-center gap-2 text-sm text-text-muted">
          <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.75} />
          Connecting to backend…
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex h-dvh w-dvw flex-col items-center justify-center gap-2 bg-bg px-6 text-center">
        <div className="heading-section text-err">Backend unavailable</div>
        <p className="max-w-md text-sm text-text-muted">
          Could not reach the SOCA API. Make sure the FastAPI server is
          running on port 8000 (or set <code className="mono">SOCA_BACKEND_URL</code>).
        </p>
      </div>
    );
  }

  if (!snapshot?.aoi_confirmed) {
    return (
      <div className="flex h-dvh w-dvw flex-col bg-bg">
        <Titlebar />
        <main className="flex flex-1 min-h-0">
          <AoiGate />
        </main>
      </div>
    );
  }

  return (
    <div className="flex h-dvh w-dvw flex-col bg-bg">
      <Titlebar />
      <main className="flex flex-1 min-h-0">
        <Sidebar />
        <section className="flex-1 min-w-0 min-h-0 hairline-r">
          <MapSection />
        </section>
        <section className="h-full w-[440px] min-w-0 bg-surface">
          <ChatPanel />
        </section>
      </main>
    </div>
  );
}
