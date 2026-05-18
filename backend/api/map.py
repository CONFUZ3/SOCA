"""Map state endpoint — view state + typed GeoJSON layers for the React map.

The frontend consumes this payload to render a rich optimisation view:

* ``layers`` — typed GeoJSON (boundary / demand / candidate / selected
  facilities / assignments / coverage rings) with per-feature properties
  (population, capacity, cost, demand_id, facility_id, …) so the map
  tooltip can surface meaningful context instead of raw indices.
* ``solution`` — a curated, variant-aware summary (status, solver,
  distance metric, objective, metrics, warnings, wall-clock time, …)
  used both for the metrics panel and for the warnings banner.

The goal is for the React layer (``map-view.tsx``) to remain a thin
renderer; any optimisation semantics live here where we still have the
full ``problem_state``.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from shapely.geometry import mapping as shapely_mapping

from backend.deps import resolve_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/map", tags=["map"])

# Max features per layer to keep payload manageable
_MAX_FEATURES = 5_000

# Solution statuses that are worth drawing on the map.  ``timeout`` and
# ``feasible`` runs still ship a partial/heuristic solution the user wants
# to inspect; only ``error`` / ``infeasible`` / missing are suppressed.
_RENDERABLE_STATUSES = {"optimal", "feasible", "timeout", "ga_fallback"}

# Common column-name candidates for enrichment of per-feature props.
_POPULATION_KEYS = (
    "population",
    "pop",
    "weight",
    "demand",
    "default_weight",
    "value",
)
_CAPACITY_KEYS = ("capacity", "cap", "max_capacity", "capacity_value")
_COST_KEYS = ("cost", "fixed_cost", "facility_cost", "opening_cost", "price")
_NAME_KEYS = ("name", "label", "id", "site_id", "place_name")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_role(name: str) -> str:
    n = name.lower()
    if n.startswith("boundary"):
        return "boundary"
    if n.startswith("demand") or "population" in n:
        return "demand"
    if "candidate" in n or "facilit" in n or "generated" in n:
        return "candidate"
    return "other"


def _first_numeric(row: Any, keys: Tuple[str, ...]) -> Optional[float]:
    """Return the first numeric value present under one of ``keys``.

    GeoPandas rows behave like dicts but raise ``KeyError`` on missing
    columns; we swallow and move on so enrichment is best-effort.
    """
    for key in keys:
        try:
            val = row[key]
        except Exception:
            continue
        if val is None:
            continue
        try:
            fval = float(val)
            if math.isnan(fval):
                continue
            return fval
        except (TypeError, ValueError):
            continue
    return None


def _first_text(row: Any, keys: Tuple[str, ...]) -> Optional[str]:
    for key in keys:
        try:
            val = row[key]
        except Exception:
            continue
        if val is None:
            continue
        try:
            s = str(val).strip()
        except Exception:
            continue
        if s:
            return s
    return None


def _enrich_feature_props(feature: Dict[str, Any], row: Any, role: str) -> None:
    """In-place attach curated numeric props for the map tooltip."""
    props = feature.setdefault("properties", {}) or {}

    name = _first_text(row, _NAME_KEYS)
    if name and "name" not in props:
        props["name"] = name

    if role == "demand":
        pop = _first_numeric(row, _POPULATION_KEYS)
        if pop is not None:
            props["population"] = pop
    elif role == "candidate":
        cap = _first_numeric(row, _CAPACITY_KEYS)
        if cap is not None:
            props["capacity"] = cap
        cost = _first_numeric(row, _COST_KEYS)
        if cost is not None:
            props["cost"] = cost

    feature["properties"] = props


def _to_geojson(
    gdf, role: str, max_features: int = _MAX_FEATURES
) -> Optional[Dict[str, Any]]:
    """Convert a GeoDataFrame to GeoJSON in EPSG:4326 with enriched props."""
    try:
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        if len(gdf) > max_features:
            gdf = gdf.sample(max_features, random_state=42)
        gj = json.loads(gdf.to_json())
        # Enrich feature props from the source row so weight / population
        # / capacity / cost are available to the client without a second
        # lookup.  We iterate by positional index because the JSON export
        # preserves row order.
        try:
            features = gj.get("features") or []
            rows = list(gdf.itertuples(index=False))
            for i, feat in enumerate(features):
                if i >= len(rows):
                    break
                _enrich_feature_props(feat, rows[i]._asdict(), role)
        except Exception as enrich_exc:
            logger.debug(
                "map _to_geojson enrichment skipped for role=%s: %s",
                role,
                enrich_exc,
            )
        return gj
    except Exception as exc:
        logger.warning("_to_geojson failed for role=%s: %s", role, exc)
        return None


def _view_from_bounds(bounds) -> Dict[str, float]:
    minx, miny, maxx, maxy = bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    extent = max(maxx - minx, maxy - miny)
    zoom = max(8.0, min(14.0, 11.0 - math.log2(max(extent, 0.001))))
    return {"longitude": cx, "latitude": cy, "zoom": zoom}


def _to_4326(gdf):
    if gdf is None:
        return None
    try:
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            return gdf.to_crs("EPSG:4326")
    except Exception:
        pass
    return gdf


def _find_candidate_gdf(data: Dict[str, Any]):
    for key in ("candidate_sites", "generated_candidates"):
        if key in data:
            return data[key]
    for key in data:
        if _classify_role(key) == "candidate":
            return data[key]
    return None


def _find_demand_gdf(data: Dict[str, Any]):
    for key in data:
        if _classify_role(key) == "demand":
            return data[key]
    return None


def _coverage_rings(sel_gdf, radius_m: float) -> Optional[Dict[str, Any]]:
    """Buffer selected facilities by ``radius_m`` to draw coverage rings.

    We project to a local equidistant CRS (Azimuthal Equidistant around
    the centroid) so the buffer distance is in metres, then project back
    to EPSG:4326 for GeoJSON output.  A failure here is logged and
    silently skipped so we never block the rest of the map response.
    """
    try:
        if sel_gdf is None or len(sel_gdf) == 0 or radius_m <= 0:
            return None
        g4326 = _to_4326(sel_gdf)
        if g4326 is None or g4326.empty:
            return None
        centroid = g4326.unary_union.centroid
        aeqd = (
            f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        projected = g4326.to_crs(aeqd)
        buffered = projected.buffer(float(radius_m))
        rings = buffered.to_crs("EPSG:4326")
        # Build GeoJSON by hand so we can attach clean props.
        features = []
        for i, geom in enumerate(rings):
            if geom is None or geom.is_empty:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": shapely_mapping(geom),
                    "properties": {
                        "facility_idx": i,
                        "radius_m": float(radius_m),
                    },
                }
            )
        if not features:
            return None
        return {"type": "FeatureCollection", "features": features}
    except Exception as exc:
        logger.warning("coverage_rings failed: %s", exc)
        return None


def _build_selected_layer(
    cand_gdf, selected: List[int]
) -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
    """Return (geojson, selected_gdf_4326) so callers can reuse the gdf."""
    if cand_gdf is None or not selected:
        return None, None
    try:
        cand_4326 = _to_4326(cand_gdf)
        valid_idx = [int(i) for i in selected if 0 <= int(i) < len(cand_4326)]
        if not valid_idx:
            return None, None
        sel_gdf = cand_4326.iloc[valid_idx].copy()
        sel_gdf["facility_idx"] = valid_idx
        gj = json.loads(sel_gdf.to_json())
        # Enrich selected-facility props (capacity/cost/name).
        try:
            rows = list(sel_gdf.itertuples(index=False))
            for i, feat in enumerate(gj.get("features") or []):
                if i >= len(rows):
                    break
                row = rows[i]._asdict()
                props = feat.setdefault("properties", {}) or {}
                props["facility_idx"] = valid_idx[i]
                nm = _first_text(row, _NAME_KEYS)
                if nm:
                    props.setdefault("name", nm)
                cap = _first_numeric(row, _CAPACITY_KEYS)
                if cap is not None:
                    props["capacity"] = cap
                cost = _first_numeric(row, _COST_KEYS)
                if cost is not None:
                    props["cost"] = cost
                feat["properties"] = props
        except Exception as enrich_exc:
            logger.debug("selected_facilities enrichment skipped: %s", enrich_exc)
        if not gj.get("features"):
            return None, None
        return gj, sel_gdf
    except Exception as exc:
        logger.warning("map_state: selected_facilities failed: %s", exc)
        return None, None


def _build_assignment_lines(
    assignments: Dict[Any, Any], demand_gdf, cand_gdf
) -> Optional[Dict[str, Any]]:
    if not assignments or demand_gdf is None or cand_gdf is None:
        return None
    try:
        d4326 = _to_4326(demand_gdf)
        c4326 = _to_4326(cand_gdf)
        lines: List[Dict[str, Any]] = []
        items = list(assignments.items())[:_MAX_FEATURES]
        for d_idx, f_idx in items:
            try:
                di = int(d_idx)
                fi = int(f_idx)
                if di < 0 or di >= len(d4326) or fi < 0 or fi >= len(c4326):
                    continue
                d_pt = d4326.geometry.iloc[di]
                f_pt = c4326.geometry.iloc[fi]
                # Try to grab demand weight for stroke-width proportioning.
                demand_row = d4326.iloc[di]
                weight = _first_numeric(demand_row, _POPULATION_KEYS) or 1.0
                lines.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [d_pt.x, d_pt.y],
                                [f_pt.x, f_pt.y],
                            ],
                        },
                        "properties": {
                            "demand_idx": di,
                            "facility_idx": fi,
                            "weight": weight,
                        },
                    }
                )
            except Exception:
                continue
        if not lines:
            return None
        return {"type": "FeatureCollection", "features": lines}
    except Exception as exc:
        logger.warning("map_state: assignments failed: %s", exc)
        return None


def _build_analysis_layers(
    ps: Dict[str, Any], data: Dict[str, Any], layers: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Inject access_heatmap / facility_coverage / facility_gaps layers.

    Mutates ``layers`` in-place: removes the plain demand layer and replaces
    it with an ``access_heatmap`` layer carrying per-feature
    ``access_distance_m``; appends coverage rings around the analysed
    facility layer and a ``facility_gaps`` layer for uncovered demand
    points. Returns the analysis summary block (or ``None`` if no analysis
    is loaded).
    """
    fa = ps.get("facility_analysis")
    if not fa:
        return None

    facility_key = fa.get("facility_dataset_key")
    radius_m = float(fa.get("service_radius_m") or 0)
    distances = fa.get("per_point_distance_m") or []
    uncovered = fa.get("uncovered_demand_indices") or []
    summary = fa.get("summary") or {}

    demand_gdf = _find_demand_gdf(data)

    # Replace the plain demand layer with an access_heatmap variant.
    if demand_gdf is not None and distances:
        try:
            d4326 = _to_4326(demand_gdf)
            gj = json.loads(d4326.to_json())
            features = gj.get("features") or []
            rows = list(d4326.itertuples(index=False))
            for i, feat in enumerate(features):
                props = feat.setdefault("properties", {}) or {}
                if i < len(rows):
                    pop = _first_numeric(rows[i]._asdict(), _POPULATION_KEYS)
                    if pop is not None:
                        props["population"] = pop
                if i < len(distances):
                    try:
                        props["access_distance_m"] = float(distances[i])
                    except (TypeError, ValueError):
                        pass
                feat["properties"] = props
            # Drop the original demand layer to avoid double-rendering.
            for j in range(len(layers) - 1, -1, -1):
                if layers[j].get("role") == "demand":
                    layers.pop(j)
            layers.append(
                {"id": "access_heatmap", "role": "access_heatmap", "geojson": gj}
            )
        except Exception as exc:
            logger.warning("map_state: access_heatmap build failed: %s", exc)

    # Coverage rings around analysed facilities.
    if facility_key and facility_key in data and radius_m > 0:
        try:
            facilities_gdf = _to_4326(data[facility_key])
            rings = _coverage_rings(facilities_gdf, radius_m)
            if rings is not None:
                layers.append(
                    {
                        "id": "facility_coverage",
                        "role": "facility_coverage",
                        "geojson": rings,
                    }
                )
        except Exception as exc:
            logger.warning("map_state: facility_coverage failed: %s", exc)

    # Gap demand points.
    if demand_gdf is not None and uncovered:
        try:
            d4326 = _to_4326(demand_gdf)
            valid = [int(i) for i in uncovered if 0 <= int(i) < len(d4326)]
            if valid:
                gap_gdf = d4326.iloc[valid].copy()
                gj = json.loads(gap_gdf.to_json())
                for i, feat in enumerate(gj.get("features") or []):
                    props = feat.setdefault("properties", {}) or {}
                    props["demand_idx"] = valid[i]
                    if i < len(valid) and valid[i] < len(distances):
                        try:
                            props["access_distance_m"] = float(distances[valid[i]])
                        except (TypeError, ValueError):
                            pass
                    feat["properties"] = props
                if gj.get("features"):
                    layers.append(
                        {
                            "id": "facility_gaps",
                            "role": "facility_gaps",
                            "geojson": gj,
                        }
                    )
        except Exception as exc:
            logger.warning("map_state: facility_gaps failed: %s", exc)

    return {
        "facility_dataset_key": facility_key,
        "service_radius_m": radius_m,
        **{k: v for k, v in summary.items() if k != "spatial_breakdown"},
        "spatial_breakdown": {
            "worst_access_points": (
                summary.get("spatial_breakdown", {}).get("worst_access_points", [])
            ),
            "n_uncovered_demand_points": len(uncovered),
        },
    }


def _build_solution_summary(
    solution: Dict[str, Any], ps: Dict[str, Any]
) -> Dict[str, Any]:
    """Curated, variant-aware summary + warnings for the UI."""
    metrics = dict(solution.get("metrics") or {})
    selected = solution.get("selected_facilities") or []
    solver_details = solution.get("solver_details") or {}

    sr = solution.get("service_radius_m")
    if sr is None:
        try:
            sr = float((ps.get("parameters") or {}).get("service_radius") or 0) or None
        except Exception:
            sr = None

    return {
        "status": solution.get("status"),
        "problem_type": solution.get("problem_type") or ps.get("problem_type"),
        "variant": solution.get("variant")
        or (ps.get("parameters") or {}).get("variant"),
        "objective_value": solution.get("objective_value"),
        "metrics": metrics,
        "n_selected": len(selected),
        "solver": solution.get("solver") or solver_details.get("solver") or "unknown",
        "solver_time_seconds": solution.get("solver_time_seconds")
        or solution.get("solution_time"),
        "gap": solver_details.get("gap"),
        "distance_metric_used": solution.get("distance_metric_used"),
        "service_radius_m": sr,
        "warnings": list(solution.get("warnings") or []),
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/state")
def map_state(ctx=Depends(resolve_session)) -> JSONResponse:
    _, record = ctx
    ps = record.get("problem_state") or {}
    data = ps.get("data") or {}

    layers: List[Dict[str, Any]] = []
    view_state: Dict[str, float] = {"longitude": 0.0, "latitude": 20.0, "zoom": 2.0}
    view_set = False

    dataset_filters = ps.get("dataset_filters") or {}

    # --- Data layers (boundary, demand, candidate) ------------------------
    for name, gdf in data.items():
        # Skip internal data attached at solve time.
        if name.startswith("_"):
            continue
        try:
            role = _classify_role(name)
            active = dataset_filters.get(name)
            if active is not None and "amenity" in gdf.columns:
                active_set = {str(s) for s in active}
                gdf = gdf[gdf["amenity"].astype(str).isin(active_set)]
            gj = _to_geojson(gdf, role)
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

    # --- Solution layers --------------------------------------------------
    solution = ps.get("solution") or {}
    solution_summary: Optional[Dict[str, Any]] = None

    if solution and solution.get("status") in _RENDERABLE_STATUSES:
        selected = solution.get("selected_facilities") or []
        assignments = solution.get("assignments") or {}
        # Use the exact GDFs the solver ran against so positional indices
        # in selected_facilities/assignments are correct.  Falls back to
        # searching the raw data store when no solve has run yet.
        _sc = ps.get("_solved_candidate_sites")
        cand_gdf = _sc if _sc is not None else _find_candidate_gdf(data)
        _sd = ps.get("_solved_demand_points")
        demand_gdf = _sd if _sd is not None else _find_demand_gdf(data)

        # Selected facilities
        sel_layer_gj, sel_gdf_4326 = _build_selected_layer(cand_gdf, selected)
        if sel_layer_gj is not None:
            layers.append(
                {
                    "id": "selected_facilities",
                    "role": "selected",
                    "geojson": sel_layer_gj,
                }
            )

        # Assignment lines
        asg_gj = _build_assignment_lines(assignments, demand_gdf, cand_gdf)
        if asg_gj is not None:
            layers.append(
                {"id": "assignments", "role": "assignment", "geojson": asg_gj}
            )

        # Coverage rings (MCLP / LSCP / any run that exposes service_radius_m)
        try:
            radius_m = solution.get("service_radius_m")
            if radius_m is None:
                radius_m = (ps.get("parameters") or {}).get("service_radius")
            radius_m = float(radius_m or 0)
        except (TypeError, ValueError):
            radius_m = 0.0
        if radius_m > 0 and sel_gdf_4326 is not None:
            rings = _coverage_rings(sel_gdf_4326, radius_m)
            if rings is not None:
                layers.append(
                    {
                        "id": "coverage_rings",
                        "role": "coverage",
                        "geojson": rings,
                    }
                )

        solution_summary = _build_solution_summary(solution, ps)

    # --- Facility-analysis layers (independent of solution) ---------------
    analysis_summary = _build_analysis_layers(ps, data, layers)

    return JSONResponse(
        {
            "view_state": view_state,
            "layers": layers,
            "solution": solution_summary,
            "analysis": analysis_summary,
        }
    )
