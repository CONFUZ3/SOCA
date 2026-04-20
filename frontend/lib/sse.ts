/**
 * Typed SSE helper. Two modes:
 *
 *   (a) GET streams (EventSource) — used for /api/events/stream.
 *   (b) POST streams (fetch + ReadableStream) — used for /api/chat/stream,
 *       which must send a request body and therefore cannot use EventSource.
 *
 * Both normalise the incoming frames to `{ event, data }` pairs, where
 * `data` is already JSON-parsed.
 */

export type SseEvent<T = unknown> = { event: string; data: T };

/**
 * Subscribe to a GET SSE endpoint. Returns an unsubscribe function.
 */
export function subscribeGetSse<T = unknown>(
  url: string,
  handler: (evt: SseEvent<T>) => void,
  options: { onError?: (e: Event) => void } = {},
): () => void {
  const es = new EventSource(url, { withCredentials: true });
  const raw = (name: string) => (e: MessageEvent) => {
    let data: T;
    try {
      data = JSON.parse(e.data) as T;
    } catch {
      data = e.data as unknown as T;
    }
    handler({ event: name, data });
  };
  // Known event types we want to consume. Unknown events fall through to the
  // default `message` listener.
  for (const name of [
    "ready",
    "activity",
    "network",
    "keepalive",
  ]) {
    es.addEventListener(name, raw(name) as EventListener);
  }
  es.onmessage = raw("message") as (e: MessageEvent) => void;
  if (options.onError) es.onerror = options.onError;
  return () => es.close();
}

/**
 * Consume a POST-SSE endpoint (chat stream). The server replies with
 * `text/event-stream` even though the method is POST, so we parse the
 * chunked stream manually.
 */
export async function* streamPostSse<T = unknown>(
  url: string,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<SseEvent<T>> {
  const res = await fetch(url, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`SSE request failed: ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // Frames are delimited by a blank line.
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const parsed = parseFrame<T>(frame);
        if (parsed) yield parsed;
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }
}

function parseFrame<T>(frame: string): SseEvent<T> | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (!dataLines.length) return null;
  const joined = dataLines.join("\n");
  let data: T;
  try {
    data = JSON.parse(joined) as T;
  } catch {
    data = joined as unknown as T;
  }
  return { event, data };
}
