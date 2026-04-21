"""Ambient SSE stream — activity_log + network-prefetch events."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from backend.deps import get_bus, require_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])

HEARTBEAT_SECONDS = 15


def _sse_frame(event: str, payload: Dict[str, Any]) -> bytes:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(payload, default=str)}\n\n"
    ).encode("utf-8")


@router.get("/stream")
async def stream_events(
    request: Request,
    ctx=Depends(require_session),
) -> StreamingResponse:
    session_id, record = ctx
    bus = get_bus()
    queue = bus.subscribe(session_id)

    async def generator() -> AsyncIterator[bytes]:
        # Replay the current activity buffer so late subscribers see recent
        # context without waiting for the next event.
        initial = record.get("_activity_log") or []
        for evt in initial[-20:]:
            yield _sse_frame("activity", evt)

        net_status = record.get("_network_status")
        if net_status:
            yield _sse_frame(
                "network",
                {
                    "status": net_status,
                    "error": record.get("_network_status_error"),
                    "stats": record.get("_network_status_stats") or {},
                },
            )

        yield _sse_frame("ready", {"session_id_present": True})

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    bus_event = await asyncio.wait_for(
                        queue.get(), timeout=HEARTBEAT_SECONDS
                    )
                    yield _sse_frame(bus_event.kind, bus_event.payload)
                except asyncio.TimeoutError:
                    yield b": keep-alive\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            bus.unsubscribe(session_id, queue)
            logger.debug("Event stream closed for session %s", session_id[:8])

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
