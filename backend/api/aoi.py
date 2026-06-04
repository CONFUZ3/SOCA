"""AOI lifecycle — geocode suggestions, boundary resolution, confirmation."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.deps import get_bus, resolve_session
from backend.services.event_bus import EventBus
from utils.activity_log import log_event
from utils.geocoder import GeocodeCandidate, suggest as geocoder_suggest
from utils.network_manager import (
    NETWORK_STATUS_ERROR_KEY,
    NETWORK_STATUS_KEY,
    NETWORK_STATUS_STATS_KEY,
    NetworkManager,
    prefetch_network_graph,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/aoi", tags=["aoi"])


# ---------------------------------------------------------------------------
# GET /api/aoi/suggest
# ---------------------------------------------------------------------------


@router.get("/suggest")
def suggest_aoi(q: str, limit: int = 6, ctx=Depends(resolve_session)):
    session_id, _ = ctx
    bus = get_bus()
    bus.bind_session(session_id)
    try:
        candidates = geocoder_suggest(q, limit=max(1, min(limit, 12)))
    finally:
        bus.bind_session(None)
    return {
        "query": q,
        "candidates": [asdict(c) for c in candidates],
    }


# ---------------------------------------------------------------------------
# POST /api/aoi/resolve — osm relation → real polygon via DataFetcher
# ---------------------------------------------------------------------------


class ResolveRequest(BaseModel):
    candidate: Dict[str, Any] = Field(
        ...,
        description="A GeocodeCandidate dict previously returned by /api/aoi/suggest.",
    )


@router.post("/resolve")
def resolve_aoi(
    body: ResolveRequest,
    ctx=Depends(resolve_session),
):
    """Resolve a geocode candidate to its boundary polygon (GeoJSON)."""
    from utils.data_fetcher import DataFetcher
    from utils.geocoder import resolve as resolve_candidate
    from utils.aoi_selector import _area_km2

    session_id, record = ctx
    bus = get_bus()
    bus.bind_session(session_id)
    try:
        cand: GeocodeCandidate = resolve_candidate(body.candidate)
        fetcher: DataFetcher = record.get("_data_fetcher") or DataFetcher()
        record["_data_fetcher"] = fetcher
        try:
            gdf = fetcher.fetch_boundaries(cand.display_name, hint=cand)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Boundary lookup failed: {exc}",
            )
        if gdf is None or len(gdf) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No boundary polygon found for this candidate.",
            )
        geom = gdf.geometry.iloc[0]
        area_km2 = _area_km2(geom)
        import json
        feature_collection = json.loads(gdf.to_json())
        return {
            "name": cand.short_name or cand.display_name,
            "display_name": cand.display_name,
            "source": gdf.attrs.get("source") or "osm",
            "area_km2": area_km2,
            "geojson": feature_collection,
        }
    finally:
        bus.bind_session(None)


# ---------------------------------------------------------------------------
# POST /api/aoi/from-dataset — derive an AOI from an uploaded dataset
# ---------------------------------------------------------------------------


class FromDatasetRequest(BaseModel):
    name: str = Field(
        ...,
        description="Dataset key in problem_state['data'] (a prior /api/data/upload).",
    )
    margin_pct: float = Field(
        0.05,
        description="Outline padding as a fraction of the larger bbox side.",
    )


@router.post("/from-dataset")
def aoi_from_dataset(
    body: FromDatasetRequest,
    ctx=Depends(resolve_session),
):
    """Derive a candidate AOI polygon from an already-uploaded dataset.

    Point/line geometry → padded convex hull; polygon geometry → dissolved
    outline used directly. Returns the same shape as ``/api/aoi/resolve`` so the
    frontend can hand it straight to the editable map + ``/api/aoi/confirm``.
    """
    import json

    import geopandas as gpd
    from shapely.ops import unary_union

    from config.settings import settings
    from utils.aoi_selector import _validate

    crs_standard = settings.CRS_STANDARD
    crs_projected = settings.CRS_PROJECTED

    _session_id, record = ctx
    ps = record["problem_state"]
    gdf = (ps.get("data") or {}).get(body.name)
    if gdf is None or len(gdf) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No dataset named {body.name!r} in session.",
        )

    geom_types = set(gdf.geometry.geom_type.unique())
    is_polygonal = geom_types <= {"Polygon", "MultiPolygon"}

    if is_polygonal:
        geom = unary_union(gdf.geometry.values)
    else:
        # Padded convex hull, computed in a metric CRS so the margin + floor
        # are in meters. The hull wraps the actual point/line extent tightly,
        # avoiding the empty-corner inflation a bounding box produces for
        # irregular shapes. A buffer handles single-point / collinear uploads
        # (hull degenerates to a point/line) gracefully.
        proj = gdf.to_crs(crs_projected)
        minx, miny, maxx, maxy = proj.total_bounds
        pad = max(body.margin_pct * max(maxx - minx, maxy - miny), 500.0)
        hull = unary_union(proj.geometry.values).convex_hull
        padded = hull.buffer(pad)
        geom = (
            gpd.GeoSeries([padded], crs=crs_projected)
            .to_crs(crs_standard)
            .iloc[0]
        )

    ok, err, area = _validate(geom, check_min_area=False)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err or "Could not derive a valid AOI from this dataset.",
        )

    out = gpd.GeoDataFrame({"name": ["aoi"], "geometry": [geom]}, crs=crs_standard)
    label = ("Boundary from " if is_polygonal else "Outline from ") + body.name
    return {
        "name": label,
        "display_name": body.name,
        "source": "upload",
        "derived": "boundary" if is_polygonal else "hull",
        "area_km2": area,
        "geojson": json.loads(out.to_json()),
    }


# ---------------------------------------------------------------------------
# POST /api/aoi/confirm — commit AOI, kick road-network prefetch
# ---------------------------------------------------------------------------


class ConfirmRequest(BaseModel):
    name: str
    source: str = "user"
    geojson: Dict[str, Any] = Field(
        ...,
        description="A GeoJSON FeatureCollection / Feature / Geometry for the AOI polygon.",
    )


def _geojson_to_boundary_gdf(gj: Dict[str, Any]):
    import geopandas as gpd
    from shapely.geometry import shape

    if gj.get("type") == "FeatureCollection":
        feats = gj.get("features") or []
        if not feats:
            raise ValueError("Empty FeatureCollection.")
        geom = shape(feats[0]["geometry"])
    elif gj.get("type") == "Feature":
        geom = shape(gj["geometry"])
    else:
        geom = shape(gj)
    return gpd.GeoDataFrame(
        {"name": ["aoi"], "geometry": [geom]}, crs="EPSG:4326"
    )


def _launch_prefetch_for_session(
    *,
    session_id: str,
    record: Dict[str, Any],
    aoi_gdf,
    bus: EventBus,
) -> None:
    """Launch road-graph prefetch bound to this session's event stream."""
    nm = record.get("_network_manager")
    if nm is None:
        # Session record should have been initialised with a NetworkManager
        # in session_store._fresh_record(); only hit this path on very old
        # records or in tests that build records by hand.
        nm = NetworkManager()
        record["_network_manager"] = nm

    # A lightweight mapping that prefetch_network_graph can write to;
    # we mirror updates into the session record + publish SSE events.
    class _SessionStateMirror(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            record[key] = value
            if key == NETWORK_STATUS_KEY:
                bus.publish_network_status(
                    session_id,
                    value or "idle",
                    error=record.get(NETWORK_STATUS_ERROR_KEY),
                    stats=record.get(NETWORK_STATUS_STATS_KEY),
                )

    mirror = _SessionStateMirror()

    def _worker():
        bus.bind_session(session_id)
        try:
            prefetch_network_graph(nm, aoi_gdf, session_state=mirror)
        finally:
            bus.bind_session(None)

    t = threading.Thread(
        target=_worker,
        name=f"soca-network-prefetch-{session_id[:8]}",
        daemon=True,
    )
    t.start()


@router.post("/confirm")
def confirm_aoi(
    body: ConfirmRequest,
    ctx=Depends(resolve_session),
):
    from utils.aoi_selector import _validate

    session_id, record = ctx
    bus = get_bus()
    bus.bind_session(session_id)

    try:
        aoi_gdf = _geojson_to_boundary_gdf(body.geojson)
        geom = aoi_gdf.geometry.iloc[0]
        ok, err, area = _validate(geom)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=err or "Invalid AOI polygon.",
            )

        ps = record["problem_state"]
        ps["aoi"] = {
            "name": body.name,
            "source": body.source,
            "area_km2": area,
            "geometry": body.geojson,
        }
        ps["aoi_confirmed"] = True
        ps["data"]["boundary_aoi"] = aoi_gdf

        # Seed a welcome message — parity with app.py initial assistant turn.
        if not record.get("messages"):
            record["messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        f"Area of interest set: **{body.name}** "
                        f"({area:,.1f} km²). Describe what you want to "
                        "optimize — for example, *place 5 clinics to maximize "
                        "coverage within 2 km*."
                    ),
                }
            ]

        log_event(
            "aoi.confirm",
            "ok",
            detail=f"{body.name} · {area:.1f} km²",
            source=body.source,
        )

        # Kick road-network prefetch in the background.
        try:
            _launch_prefetch_for_session(
                session_id=session_id,
                record=record,
                aoi_gdf=aoi_gdf,
                bus=bus,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("AOI prefetch launch failed: %s", exc)

        return {
            "ok": True,
            "aoi": ps["aoi"],
            "area_km2": area,
        }
    finally:
        bus.bind_session(None)
