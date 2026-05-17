"""Dataset upload, list, fetch-as-GeoJSON, and delete."""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel

from backend.deps import get_bus, resolve_session
from utils.activity_log import log_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])


def _get_processor(record: Dict[str, Any]):
    from utils.data_processor import DataProcessor

    dp = record.get("_data_processor")
    if dp is None:
        dp = DataProcessor()
        record["_data_processor"] = dp
    return dp


def _dataset_info(
    name: str, gdf, filters: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
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
    preferred_numeric = ("population", "demand", "weight", "capacity", "cost")
    if gdf is not None:
        try:
            for col in preferred_numeric:
                if col not in columns:
                    continue
                series = gdf[col].dropna()
                if len(series) == 0:
                    continue
                numeric_preview[col] = float(series.mean())
                if len(numeric_preview) >= 2:
                    break
        except Exception:
            numeric_preview = {}

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
    subcategory_counts: Dict[str, int] = {}
    if gdf is not None and "amenity" in columns:
        try:
            counts = gdf["amenity"].dropna().astype(str).value_counts()
            subcategory_counts = {str(k): int(v) for k, v in counts.items()}
            # Sort by count desc, then alphabetically for stable order.
            available_subcategories = sorted(
                subcategory_counts.keys(),
                key=lambda k: (-subcategory_counts[k], k),
            )
        except Exception:
            available_subcategories = []
            subcategory_counts = {}

    total_features = int(len(gdf)) if gdf is not None else 0

    active_subcategories: Optional[List[str]] = None
    if filters is not None and name in filters:
        active_subcategories = filters[name]

    # Compute the post-filter feature count so the UI can show the actual
    # number of features that will be passed to the optimizer / map.
    if active_subcategories is not None and "amenity" in columns and gdf is not None:
        try:
            active_set = {str(s) for s in active_subcategories}
            mask = gdf["amenity"].astype(str).isin(active_set)
            active_num_features = int(mask.sum())
        except Exception:
            active_num_features = total_features
    else:
        active_num_features = total_features

    result: Dict[str, Any] = {
        "name": name,
        "num_features": total_features,
        "active_num_features": active_num_features,
        "geometry_type": geom_type,
        "columns": columns,
        "bounds": bounds,
        "source": gdf.attrs.get("source") if hasattr(gdf, "attrs") else None,
        "role": role,
        "source_details": source_values,
        "numeric_preview": numeric_preview,
        "available_subcategories": available_subcategories,
        "subcategory_counts": subcategory_counts,
    }
    if active_subcategories is not None:
        result["active_subcategories"] = active_subcategories
    return result


class _NamedBuffer(io.BytesIO):
    """BytesIO subclass with a ``.name`` attribute for DataProcessor."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


class _SubcategoryFilterBody(BaseModel):
    active_subcategories: List[str]


@router.get("")
def list_datasets(ctx=Depends(resolve_session)) -> Dict[str, List[Dict[str, Any]]]:
    _session_id, record = ctx
    ps = record["problem_state"]
    data = ps.get("data") or {}
    filters = ps.get("dataset_filters") or {}
    return {"datasets": [_dataset_info(n, g, filters) for n, g in data.items()]}


@router.patch("/{name}/filter")
def set_subcategory_filter(
    name: str,
    body: _SubcategoryFilterBody,
    ctx=Depends(resolve_session),
) -> Dict[str, Any]:
    _session_id, record = ctx
    ps = record["problem_state"]
    data = ps.get("data") or {}
    if name not in data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No dataset named {name!r} in session.",
        )
    filters = ps.setdefault("dataset_filters", {})
    filters[name] = body.active_subcategories
    gdf = data[name]
    return _dataset_info(name, gdf, filters)


@router.post("/upload")
async def upload_dataset(
    files: List[UploadFile] = File(...),
    ctx=Depends(resolve_session),
):
    session_id, record = ctx
    dp = _get_processor(record)
    ps = record["problem_state"]
    data_store = ps.setdefault("data", {})

    bus = get_bus()
    bus.bind_session(session_id)
    loaded: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    try:
        for up in files:
            try:
                raw = await up.read()
                buf = _NamedBuffer(raw, up.filename or "uploaded")
                gdf = dp.load_file(buf)
                gdf = dp.preprocess_data(gdf)
                data_store[up.filename] = gdf
                loaded.append(_dataset_info(up.filename, gdf))
                log_event(
                    "dataset.upload",
                    "ok",
                    detail=f"{up.filename} · {len(gdf)} features",
                    source="user",
                )
            except Exception as exc:
                logger.error("upload_dataset %s failed: %s", up.filename, exc)
                errors.append({"name": up.filename, "error": str(exc)})
                log_event(
                    "dataset.upload",
                    "fail",
                    detail=f"{up.filename}: {exc}",
                    source="user",
                )

        filters = ps.get("dataset_filters") or {}
        return {
            "loaded": loaded,
            "errors": errors,
            "datasets": [_dataset_info(n, g, filters) for n, g in data_store.items()],
        }
    finally:
        bus.bind_session(None)


@router.delete("/{name}")
def delete_dataset(name: str, ctx=Depends(resolve_session)) -> Dict[str, Any]:
    _session_id, record = ctx
    data = record["problem_state"].get("data") or {}
    if name not in data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No dataset named {name!r} in session.",
        )
    del data[name]
    return {"ok": True, "removed": name}


@router.get("/{name}.geojson")
def get_geojson(name: str, ctx=Depends(resolve_session)) -> Response:
    _session_id, record = ctx
    data = record["problem_state"].get("data") or {}
    gdf = data.get(name)
    if gdf is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No dataset named {name!r} in session.",
        )
    try:
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        pass
    payload = json.loads(gdf.to_json())
    return Response(
        content=json.dumps(payload),
        media_type="application/geo+json",
    )
