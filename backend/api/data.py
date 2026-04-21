"""Dataset upload, list, fetch-as-GeoJSON, and delete."""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)

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


def _dataset_info(name: str, gdf) -> Dict[str, Any]:
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
    return {
        "name": name,
        "num_features": int(len(gdf)) if gdf is not None else 0,
        "geometry_type": geom_type,
        "columns": columns,
        "bounds": bounds,
        "source": gdf.attrs.get("source") if hasattr(gdf, "attrs") else None,
    }


class _NamedBuffer(io.BytesIO):
    """BytesIO subclass with a ``.name`` attribute for DataProcessor."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


@router.get("")
def list_datasets(ctx=Depends(resolve_session)) -> Dict[str, List[Dict[str, Any]]]:
    _session_id, record = ctx
    data = record["problem_state"].get("data") or {}
    return {"datasets": [_dataset_info(n, g) for n, g in data.items()]}


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

        return {
            "loaded": loaded,
            "errors": errors,
            "datasets": [_dataset_info(n, g) for n, g in data_store.items()],
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
