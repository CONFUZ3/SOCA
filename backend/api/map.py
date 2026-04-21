"""Map state endpoint — view state + typed GeoJSON layers for the React map."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend.deps import resolve_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/map", tags=["map"])

# Max features per layer to keep payload manageable
_MAX_FEATURES = 5_000


def _classify_role(name: str) -> str:
    n = name.lower()
    if n.startswith("boundary"):
        return "boundary"
    if n.startswith("demand") or "population" in n:
        return "demand"
    if "candidate" in n or "facilit" in n or "generated" in n:
        return "candidate"
    return "other"


def _to_geojson(gdf, max_features: int = _MAX_FEATURES) -> Optional[Dict[str, Any]]:
    try:
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        if len(gdf) > max_features:
            gdf = gdf.sample(max_features, random_state=42)
        return json.loads(gdf.to_json())
    except Exception as exc:
        logger.warning("_to_geojson failed: %s", exc)
        return None


def _view_from_bounds(bounds) -> Dict[str, float]:
    minx, miny, maxx, maxy = bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    extent = max(maxx - minx, maxy - miny)
    zoom = max(8.0, min(14.0, 11.0 - math.log2(max(extent, 0.001))))
    return {"longitude": cx, "latitude": cy, "zoom": zoom}


@router.get("/state")
def map_state(ctx=Depends(resolve_session)) -> JSONResponse:
    _, record = ctx
    ps = record.get("problem_state") or {}
    data = ps.get("data") or {}

    layers: List[Dict[str, Any]] = []
    view_state: Dict[str, float] = {"longitude": 0.0, "latitude": 20.0, "zoom": 2.0}
    view_set = False

    # --- Data layers (boundary, demand, candidate) ---
    # Two-pass: first collect all layers, then set view_state (boundary wins).
    for name, gdf in data.items():
        try:
            role = _classify_role(name)
            gj = _to_geojson(gdf)
            if not gj or not gj.get("features"):
                continue
            layers.append({"id": name, "role": role, "geojson": gj})
        except Exception as exc:
            logger.warning("map_state: dataset %r failed: %s", name, exc)

    # Derive view_state: boundary takes priority, fall back to first layer.
    for layer in layers:
        if layer["role"] == "boundary" or not view_set:
            try:
                gdf = data[layer["id"]]
                view_state = _view_from_bounds(gdf.total_bounds)
                view_set = True
                if layer["role"] == "boundary":
                    break
            except Exception:
                pass

    # --- Solution layers (selected facilities + assignment lines) ---
    solution = ps.get("solution")
    solution_summary: Optional[Dict[str, Any]] = None

    if solution and solution.get("status") == "optimal":
        selected = solution.get("selected_facilities") or []
        assignments = solution.get("assignments") or {}

        # Find candidate GeoDataFrame
        cand_gdf = None
        for key in ("candidate_sites", "generated_candidates"):
            if key in data:
                cand_gdf = data[key]
                break
        if cand_gdf is None:
            for key in data:
                if _classify_role(key) == "candidate":
                    cand_gdf = data[key]
                    break

        # Find demand GeoDataFrame
        demand_gdf = None
        for key in data:
            if _classify_role(key) == "demand":
                demand_gdf = data[key]
                break

        # Selected facilities
        if cand_gdf is not None and selected:
            try:
                cand_4326 = cand_gdf
                if cand_4326.crs is not None and cand_4326.crs.to_epsg() != 4326:
                    cand_4326 = cand_4326.to_crs("EPSG:4326")
                valid_idx = [int(i) for i in selected if int(i) < len(cand_4326)]
                sel_gdf = cand_4326.iloc[valid_idx]
                gj = json.loads(sel_gdf.to_json())
                if gj.get("features"):
                    layers.append({"id": "selected_facilities", "role": "selected", "geojson": gj})
            except Exception as exc:
                logger.warning("map_state: selected_facilities failed: %s", exc)

        # Assignment lines
        if assignments and demand_gdf is not None and cand_gdf is not None:
            try:
                d4326 = demand_gdf
                if d4326.crs is not None and d4326.crs.to_epsg() != 4326:
                    d4326 = d4326.to_crs("EPSG:4326")
                c4326 = cand_gdf
                if c4326.crs is not None and c4326.crs.to_epsg() != 4326:
                    c4326 = c4326.to_crs("EPSG:4326")

                lines = []
                items = list(assignments.items())[:_MAX_FEATURES]
                for d_idx, f_idx in items:
                    try:
                        d_pt = d4326.geometry.iloc[int(d_idx)]
                        f_pt = c4326.geometry.iloc[int(f_idx)]
                        lines.append({
                            "type": "Feature",
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[d_pt.x, d_pt.y], [f_pt.x, f_pt.y]],
                            },
                            "properties": {"demand": int(d_idx), "facility": int(f_idx)},
                        })
                    except Exception:
                        continue

                if lines:
                    layers.append({
                        "id": "assignments",
                        "role": "assignment",
                        "geojson": {"type": "FeatureCollection", "features": lines},
                    })
            except Exception as exc:
                logger.warning("map_state: assignments failed: %s", exc)

        solution_summary = {
            "status": solution.get("status"),
            "objective_value": solution.get("objective_value"),
            "metrics": solution.get("metrics") or {},
            "n_selected": len(selected),
            "problem_type": ps.get("problem_type"),
        }

    return JSONResponse({
        "view_state": view_state,
        "layers": layers,
        "solution": solution_summary,
    })
