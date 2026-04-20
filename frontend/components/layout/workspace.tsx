"use client";

import { useSession } from "@/hooks/use-session";
import { useEventsStream } from "@/hooks/use-events-stream";
import { useStore } from "@/lib/store";
import { AoiGate } from "@/components/aoi/aoi-gate";
import { ChatPanel } from "@/components/chat/chat-panel";
import { MapPlaceholder } from "@/components/map/map-placeholder";
import { Sidebar } from "@/components/layout/sidebar";
import { Titlebar } from "@/components/layout/titlebar";
import { Loader2 } from "lucide-react";

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
          <MapPlaceholder />
        </section>
        <section className="h-full w-[440px] min-w-0 bg-surface">
          <ChatPanel />
        </section>
      </main>
    </div>
  );
}
