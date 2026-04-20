"""Road-network prefetch status + manual refresh."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.aoi import _launch_prefetch_for_session
from backend.deps import get_bus, resolve_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/network", tags=["network"])


@router.get("/status")
def network_status(ctx=Depends(resolve_session)) -> Dict[str, Any]:
    _session_id, record = ctx
    return {
        "status": record.get("_network_status"),
        "error": record.get("_network_status_error"),
        "stats": record.get("_network_status_stats"),
    }


@router.post("/refresh")
def refresh_network(ctx=Depends(resolve_session)) -> Dict[str, Any]:
    session_id, record = ctx
    ps = record.get("problem_state") or {}
    if not ps.get("aoi_confirmed"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No AOI confirmed; nothing to prefetch.",
        )
    aoi_gdf = ps.get("data", {}).get("boundary_aoi")
    if aoi_gdf is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="AOI boundary not in session.",
        )
    nm = record.get("_network_manager")
    if nm is not None:
        try:
            nm.clear_cache()
        except Exception:
            pass
    bus = get_bus()
    _launch_prefetch_for_session(
        session_id=session_id,
        record=record,
        aoi_gdf=aoi_gdf,
        bus=bus,
    )
    return {"ok": True}
