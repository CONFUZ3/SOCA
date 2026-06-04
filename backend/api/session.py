"""Session lifecycle and a readable snapshot of server-side state."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Response

from backend.deps import resolve_session, set_session_cookie
from backend.services.session_store import get_default_store

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
    columns = [c for c in getattr(gdf, "columns", []) if c != "geometry"]
    source_values: List[str] = []
    if gdf is not None and "data_source" in columns:
        try:
            source_values = [
                str(v)
                for v in gdf["data_source"].dropna().astype(str).unique().tolist()[:3]
            ]
        except Exception:
            source_values = []

    numeric_preview: Dict[str, float] = {}
    numeric_summary: List[Dict[str, Any]] = []
    preferred_numeric = ("population", "demand", "weight", "capacity", "cost")
    if gdf is not None:
        try:
            import pandas as pd

            for col in preferred_numeric:
                if col not in columns:
                    continue
                series = pd.to_numeric(gdf[col], errors="coerce").dropna()
                if len(series) == 0:
                    continue
                numeric_preview[col] = float(series.mean())
                if col in ("population", "demand", "weight"):
                    numeric_summary.append(
                        {
                            "key": col,
                            "label": f"total {col}",
                            "value": float(series.sum()),
                            "stat": "total",
                        }
                    )
                else:
                    numeric_summary.append(
                        {
                            "key": col,
                            "label": f"avg {col}",
                            "value": float(series.mean()),
                            "stat": "mean",
                        }
                    )
                if len(numeric_preview) >= 2:
                    break
        except Exception:
            numeric_preview = {}
            numeric_summary = []

    lname = name.lower()
    if lname.startswith("boundary"):
        role = "boundary"
    elif lname.startswith("demand") or "population" in lname:
        role = "demand"
    elif "candidate" in lname or "facilit" in lname or "generated" in lname:
        role = "candidate"
    else:
        role = "other"

    available_subcategories: List[str] = []
    if gdf is not None and "amenity" in columns:
        try:
            available_subcategories = sorted(
                str(v) for v in gdf["amenity"].dropna().unique().tolist()
            )
        except Exception:
            available_subcategories = []

    result: dict = {
        "name": name,
        "num_features": int(len(gdf)) if gdf is not None else 0,
        "geometry_type": geom_type,
        "columns": columns,
        "bounds": bounds,
        "source": gdf.attrs.get("source") if hasattr(gdf, "attrs") else None,
        "role": role,
        "source_details": source_values,
        "numeric_preview": numeric_preview,
        "numeric_summary": numeric_summary,
        "available_subcategories": available_subcategories,
    }
    return result


def _datasets_from_problem_state(ps: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = ps.get("data") or {}
    filters = ps.get("dataset_filters") or {}
    summaries = []
    for name, gdf in data.items():
        s = _dataset_summary(name, gdf)
        if name in filters:
            s["active_subcategories"] = filters[name]
        summaries.append(s)
    return summaries


def _snapshot(record: Dict[str, Any]) -> Dict[str, Any]:
    ps = record.get("problem_state") or {}
    aoi = ps.get("aoi")
    solution_history = ps.get("solution_history") or []
    try:
        solution_version = int(len(solution_history))
    except Exception:
        solution_version = 0
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
        "solution_version": solution_version,
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


@router.delete("")
def reset_session(response: Response, ctx=Depends(resolve_session)):
    """Drop the current session and return a fresh one with the same cookie."""
    session_id, _ = ctx
    store = get_default_store()
    store.drop(session_id)
    _, fresh_record = store.get_or_create(session_id)
    set_session_cookie(response, session_id)
    return {"session_id_visible": False, **_snapshot(fresh_record)}
