"use client";

import { useEffect, useRef } from "react";
import { useStore } from "@/lib/store";
import { useChat } from "@/hooks/use-chat";
import { Composer } from "./composer";
import { Message } from "./message";
import { ActivityList } from "@/components/activity/activity-list";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";

export function ChatPanel() {
  const turns = useStore((s) => s.turns);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const { send, cancel, busy } = useChat(() => {
    // nothing after turn — the store handles it.
  });

  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [turns]);

  return (
    <section className="flex h-full w-full flex-col">
      <Tabs defaultValue="chat" className="flex h-full min-h-0 flex-col">
        <div className="flex items-center justify-between px-3 pt-2.5 pb-2">
          <TabsList>
            <TabsTrigger value="chat">Messages</TabsTrigger>
            <TabsTrigger value="activity">Activity</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent
          value="chat"
          className="flex min-h-0 flex-1 flex-col focus:outline-none"
        >
          <div
            ref={scrollerRef}
            className="flex-1 overflow-y-auto px-3 pb-2"
          >
            {turns.length === 0 ? (
              <EmptyChat />
            ) : (
              <div className="mx-auto max-w-3xl">
                {turns.map((t) => (
                  <Message key={t.id} turn={t} />
                ))}
              </div>
            )}
          </div>
          <div className="border-t border-border bg-surface/80 px-3 py-2 backdrop-blur">
            <div className="mx-auto max-w-3xl">
              <Composer onSend={send} onCancel={cancel} busy={busy} />
            </div>
          </div>
        </TabsContent>

        <TabsContent
          value="activity"
          className="min-h-0 flex-1 overflow-y-auto px-3 pb-3 focus:outline-none"
        >
          <ActivityList />
        </TabsContent>
      </Tabs>
    </section>
  );
}

function EmptyChat() {
  return (
    <div className="mx-auto mt-6 max-w-md px-4">
      <div className="heading-section">Start a conversation</div>
      <p className="mt-1.5 text-sm text-text-muted">
        Describe the problem in plain English. SOCA picks the right
        optimisation model, fetches data, and shows every step — the map
        updates as tools run.
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
