"""Chat endpoint — streams ADK agent events to the client as SSE."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.soca_agent import SOCAAgent
from backend.deps import get_bus, resolve_session
from solvers.registry import problem_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


def _ensure_agent(record: Dict[str, Any]) -> SOCAAgent:
    agent = record.get("_soca_agent")
    if agent is not None:
        return agent
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "GEMINI_API_KEY is not configured. Set it in the server "
                "environment before using the chat endpoint."
            ),
        )
    agent = SOCAAgent(api_key=api_key, problem_registry=problem_registry)
    record["_soca_agent"] = agent
    return agent


def _build_data_summary(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from utils.data_processor import DataProcessor

    ps = record["problem_state"]
    data = ps.get("data") or {}
    if not data:
        return None
    dp = record.get("_data_processor") or DataProcessor()
    record["_data_processor"] = dp

    summary: Dict[str, Any] = {}
    for name, gdf in data.items():
        try:
            dtypes = {c: str(gdf[c].dtype) for c in gdf.columns if c != "geometry"}
        except Exception:
            dtypes = {}
        try:
            sample_values: Dict[str, Any] = {}
            column_stats: Dict[str, Any] = {}
            for col in gdf.columns:
                if col.lower() in ("geometry", "shape"):
                    continue
                non_null = gdf[col].dropna()
                if len(non_null) > 0:
                    sample_values[col] = non_null.iloc[0]
                    if gdf[col].dtype in ("int64", "float64", "int32", "float32"):
                        column_stats[col] = {
                            "mean": float(gdf[col].mean()),
                            "max": float(gdf[col].max()),
                        }
        except Exception:
            sample_values, column_stats = {}, {}

        summary[name] = {
            "num_features": len(gdf),
            "geometry_type": (
                gdf.geometry.type.unique()[0] if len(gdf) > 0 else "Unknown"
            ),
            "columns": [c for c in gdf.columns if c != "geometry"],
            "dtypes": dtypes,
            "bounds": gdf.total_bounds.tolist() if len(gdf) > 0 else [],
            "capacity_columns": dp.identify_capacity_columns(gdf),
            "cost_columns": dp.identify_cost_columns(gdf),
            "demand_columns": dp.identify_demand_columns(gdf),
            "sample_values": sample_values,
            "column_stats": column_stats,
        }
    return summary


def _sse(kind: str, payload: Dict[str, Any]) -> bytes:
    return (
        f"event: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"
    ).encode("utf-8")


@router.post("/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    ctx=Depends(resolve_session),
) -> StreamingResponse:
    session_id, record = ctx
    agent = _ensure_agent(record)
    bus = get_bus()

    ps = record["problem_state"]
    ps["_network_manager"] = record.get("_network_manager")
    ps["_generated_sites_count"] = record.get("generated_sites_count", 100)
    ps["_generated_sites_seed"] = record.get("generated_sites_seed")

    history = list(record.get("messages") or [])
    record.setdefault("messages", []).append(
        {"role": "user", "content": body.message}
    )

    data_summary = _build_data_summary(record)

    async def generator() -> AsyncIterator[bytes]:
        bus.bind_session(session_id)
        final_text = ""
        tool_calls: List[str] = []
        try:
            yield _sse("start", {"ok": True})
            async for ev in agent.chat_stream(
                user_message=body.message,
                conversation_history=history,
                problem_state=ps,
                uploaded_data_summary=data_summary,
            ):
                if await request.is_disconnected():
                    break

                kind = ev.get("type")
                if kind == "tool_call_start":
                    yield _sse(
                        "tool_call_start",
                        {"name": ev.get("name"), "args": ev.get("args") or {}},
                    )
                elif kind == "tool_call_result":
                    yield _sse(
                        "tool_call_result",
                        {"name": ev.get("name"), "summary": ev.get("summary")},
                    )
                elif kind == "token":
                    yield _sse("token", {"text": ev.get("text") or ""})
                elif kind == "final":
                    final_text = ev.get("text") or ""
                    tool_calls = list(ev.get("tool_calls") or [])
                    yield _sse(
                        "final",
                        {"text": final_text, "tool_calls": tool_calls},
                    )
                elif kind == "error":
                    yield _sse("error", {"message": ev.get("message") or ""})

            if final_text:
                record["messages"].append(
                    {
                        "role": "assistant",
                        "content": final_text,
                        "tool_calls": tool_calls,
                    }
                )
        finally:
            bus.bind_session(None)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history")
def history(ctx=Depends(resolve_session)) -> Dict[str, Any]:
    _session_id, record = ctx
    return {"messages": record.get("messages") or []}
