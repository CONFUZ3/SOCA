"""Per-session event bus fed by ``utils.activity_log`` sinks.

A single module-level registry maps ``session_id`` → list[asyncio.Queue].
Each ``GET /api/events/stream`` handler registers a queue on entry and
drops it on disconnect. A global ``activity_log`` sink routes each event
into every queue belonging to the "current" session.

Session resolution for log events is thread-local: before the ADK runner
is invoked (or a background prefetch runs), the handler calls
``bind_session(session_id)`` which installs the id on the current thread.
Anything logged on that thread is fanned out to that session's queues
only. This mirrors how ``state_bridge`` already scopes thread-local data.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from utils.activity_log import (
    ActivityEvent,
    event_to_dict,
    register_sink,
)

logger = logging.getLogger(__name__)


@dataclass
class BusEvent:
    """Envelope used on the wire (SSE JSON payload)."""

    kind: str           # "activity" | "network" | "heartbeat" | "state"
    payload: Dict[str, Any]

    def to_sse(self) -> str:
        import json
        return f"event: {self.kind}\ndata: {json.dumps(self.payload)}\n\n"


class _ThreadBinding:
    """Thread-local holder for the current session id used by the sink."""

    def __init__(self) -> None:
        self._local = threading.local()

    def get(self) -> Optional[str]:
        return getattr(self._local, "session_id", None)

    def set(self, session_id: Optional[str]) -> None:
        self._local.session_id = session_id


class EventBus:
    """Session-keyed fan-out of activity-log + network-status events."""

    def __init__(self) -> None:
        self._queues: Dict[str, list[asyncio.Queue]] = {}
        self._lock = threading.RLock()
        self._binding = _ThreadBinding()
        self._sink_unregister = register_sink(self._on_activity_event)

    # ------------------------------------------------------------------
    # Thread-local session scoping
    # ------------------------------------------------------------------

    def bind_session(self, session_id: Optional[str]) -> None:
        self._binding.set(session_id)

    def current_session(self) -> Optional[str]:
        return self._binding.get()

    # ------------------------------------------------------------------
    # Subscriber API (used by /api/events/stream)
    # ------------------------------------------------------------------

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """Register a new queue for ``session_id`` and return it."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        with self._lock:
            self._queues.setdefault(session_id, []).append(queue)
        logger.debug("EventBus: +1 subscriber for %s…", session_id[:8])
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            queues = self._queues.get(session_id)
            if not queues:
                return
            try:
                queues.remove(queue)
            except ValueError:
                pass
            if not queues:
                self._queues.pop(session_id, None)

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(
        self, session_id: str, kind: str, payload: Dict[str, Any]
    ) -> None:
        """Push a ``BusEvent`` into every queue for ``session_id``."""
        event = BusEvent(kind=kind, payload=payload)
        with self._lock:
            queues = list(self._queues.get(session_id, []))
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop-oldest policy keeps streams responsive on slow clients.
                try:
                    _ = q.get_nowait()
                    q.put_nowait(event)
                except Exception:  # pragma: no cover - defensive
                    pass

    def publish_network_status(
        self,
        session_id: str,
        status: str,
        *,
        error: Optional[str] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.publish(
            session_id,
            "network",
            {"status": status, "error": error, "stats": stats or {}},
        )

    # ------------------------------------------------------------------
    # activity_log sink
    # ------------------------------------------------------------------

    def _on_activity_event(self, evt: ActivityEvent) -> None:
        sid = self.current_session()
        if not sid:
            return  # event produced outside a bound session — ignore
        payload = event_to_dict(evt)
        self.publish(sid, "activity", payload)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Unregister the activity-log sink (primarily for tests)."""
        try:
            self._sink_unregister()
        except Exception:  # pragma: no cover
            pass
        with self._lock:
            self._queues.clear()


_default_bus: Optional[EventBus] = None


def get_default_bus() -> EventBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


__all__ = [
    "BusEvent",
    "EventBus",
    "get_default_bus",
]
