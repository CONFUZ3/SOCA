"""Session lifecycle and a readable snapshot of server-side state."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Response

from backend.deps import resolve_session, set_session_cookie

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["session"])


def _dataset_summary(name: str, gdf) -> Dict[str, Any]:
    geom_type = "Unknown"
    try:
        if len(gdf) > 0:
            geom_type = str(gdf.geometry.type.unique()[0])
    except Exception:
        pass
    try:
        bounds = gdf.total_bounds.tolist() if len(gdf) > 0 else []
    except Exception:
        bounds = []
    return {
        "name": name,
        "num_features": int(len(gdf)) if gdf is not None else 0,
        "geometry_type": geom_type,
        "columns": [c for c in getattr(gdf, "columns", []) if c != "geometry"],
        "bounds": bounds,
        "source": gdf.attrs.get("source") if hasattr(gdf, "attrs") else None,
    }


def _datasets_from_problem_state(ps: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = ps.get("data") or {}
    return [_dataset_summary(name, gdf) for name, gdf in data.items()]


def _snapshot(record: Dict[str, Any]) -> Dict[str, Any]:
    ps = record.get("problem_state") or {}
    aoi = ps.get("aoi")
    return {
        "session_id_present": True,
        "aoi": aoi,
        "aoi_confirmed": bool(ps.get("aoi_confirmed")),
        "problem_type": ps.get("problem_type"),
        "parameters": ps.get("parameters") or {},
        "constraints": ps.get("constraints") or {},
        "datasets": _datasets_from_problem_state(ps),
        "has_solution": ps.get("solution") is not None,
        "solution_status": (ps.get("solution") or {}).get("status"),
        "messages": record.get("messages") or [],
        "network": {
            "status": record.get("_network_status"),
            "error": record.get("_network_status_error"),
            "stats": record.get("_network_status_stats"),
        },
        "settings": {
            "generated_sites_count": record.get("generated_sites_count", 100),
            "generated_sites_seed": record.get("generated_sites_seed"),
        },
    }


@router.post("")
def create_or_refresh_session(
    response: Response,
    ctx=Depends(resolve_session),
):
    """Create a session cookie if missing, then return a snapshot.

    Idempotent: repeated calls return the same session record + snapshot.
    """
    session_id, record = ctx
    set_session_cookie(response, session_id)
    return {"session_id_visible": False, **_snapshot(record)}


@router.get("")
def get_session(ctx=Depends(resolve_session)):
    _session_id, record = ctx
    return _snapshot(record)
