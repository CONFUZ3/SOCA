"use client";

import { useCallback, useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useStore } from "@/lib/store";
import { useChat } from "@/hooks/use-chat";
import { useSession } from "@/hooks/use-session";
import { Composer } from "./composer";
import { Message } from "./message";
import { ActivityGroupCard } from "./activity-group-card";
import { NetworkStatusStrip } from "./network-status-strip";
import { SubcategoryPicker } from "./subcategory-picker";

export function ChatPanel() {
  const items = useStore((s) => s.items);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const qc = useQueryClient();
  const { refresh } = useSession();
  const handleMapUpdate = useCallback(
    () => qc.invalidateQueries({ queryKey: ["map-state"] }),
    [qc],
  );
  const { send, cancel, busy } = useChat(refresh, handleMapUpdate);

  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [items]);

  return (
    <section className="flex h-full w-full flex-col">
      <NetworkStatusStrip />

      <div
        ref={scrollerRef}
        className="flex-1 min-h-0 overflow-y-auto px-3 pb-2 pt-2"
      >
        {items.length === 0 ? (
          <EmptyChat />
        ) : (
          <div className="mx-auto max-w-3xl">
            {items.map((item) =>
              item.kind === "turn" ? (
                <Message key={item.id} turn={item} />
              ) : (
                <ActivityGroupCard key={item.id} group={item} />
              ),
            )}
          </div>
        )}
      </div>

      <div className="border-t border-border bg-surface/80 px-3 py-2 backdrop-blur">
        <div className="mx-auto max-w-3xl space-y-2">
          <SubcategoryPicker />
          <Composer onSend={send} onCancel={cancel} busy={busy} />
        </div>
      </div>
    </section>
  );
}

function EmptyChat() {
  return (
    <div className="mx-auto mt-6 max-w-md px-4">
      <div className="heading-section">Start a conversation</div>
      <p className="mt-1.5 text-sm text-text-muted">
        Describe the problem in plain English. SOCA picks the right
        optimisation model, fetches data, and shows every step inline —
        sources, timing, and progress all appear here as it works.
      </p>
      <ul className="mt-4 space-y-1.5 text-sm text-text">
        <li className="rounded border border-border bg-surface px-2.5 py-1.5">
          Place 5 hospitals in Nairobi to minimise travel distance.
        </li>
        <li className="rounded border border-border bg-surface px-2.5 py-1.5">
          Maximise clinic coverage within 2 km using 4 facilities.
        </li>
        <li className="rounded border border-border bg-surface px-2.5 py-1.5">
          How many fire stations do I need to cover every block within 3 km?
        </li>
      </ul>
    </div>
  );
}
