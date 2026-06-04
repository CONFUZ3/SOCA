"""ADK tool: read-only diagnostic analysis of existing facilities.

Lets users ask "where are schools lacking?" or "what's access to hospitals
like?" without running an optimization. Reuses the already-fetched
demand grid + facility POIs + cached road graph; never re-fetches.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from google.adk.tools.tool_context import ToolContext

from .state_bridge import (
    get_data,
    get_network_manager,
    get_problem_state,
)

logger = logging.getLogger(__name__)


_UNIT_TO_METRES = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    "nm": 1852.0, "nmi": 1852.0,
}


def _find_facility_key(data_store: dict, explicit_key: Optional[str]) -> tuple:
    """Return (key, error_or_None). Auto-picks the only matching layer."""
    if explicit_key:
        if explicit_key not in data_store:
            return None, (
                f"Dataset '{explicit_key}' not found. "
                f"Available: {sorted(data_store.keys())}"
            )
        return explicit_key, None

    matches = [k for k in data_store if "_facilities_" in k.lower()]
    if not matches:
        return None, (
            "No facility dataset loaded. Call fetch_city_data() with a "
            "poi_category first (e.g. 'health', 'education')."
        )
    if len(matches) > 1:
        return None, (
            f"Multiple facility datasets loaded: {matches}. "
            "Pass facility_dataset_key='<one of these>' to disambiguate."
        )
    return matches[0], None


def _find_demand_gdf(data_store: dict):
    for name, gdf in data_store.items():
        if name.lower().startswith("demand_") or name.lower() == "demand_points":
            return gdf
    return None


def _find_boundary_gdf(data_store: dict):
    for name, gdf in data_store.items():
        if name.lower().startswith("boundary"):
            return gdf
    return None


def analyze_existing_facilities(
    facility_dataset_key: Optional[str] = None,
    service_radius: float = 5.0,
    service_radius_unit: str = "km",
    distance_metric: str = "network",
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Analyze coverage and access for already-fetched facilities.

    Use this for read-only diagnostic questions about existing facilities
    (schools, hospitals, etc.) the app has fetched — e.g. "what areas lack
    schools?", "average distance to the nearest hospital?", "facility
    density per km²?". Does NOT run an optimization.

    Args:
        facility_dataset_key: Optional key of the facility GeoDataFrame in
            the data store. If omitted, auto-picks the only ``*_facilities_*``
            layer; errors when zero or multiple match.
        service_radius: Coverage radius value (default 5).
        service_radius_unit: Unit for service_radius. One of m, km, miles,
            ft, yd, nm. Default "km".
        distance_metric: "network" (road shortest-path, default) or
            "euclidean" (geodesic straight-line). Falls back to euclidean
            with a warning if the road graph is unavailable.

    Returns:
        dict with keys:
          coverage: pct_demand_covered, pct_points_covered,
                    uncovered_demand_weight, service_radius_m
          access:   avg_distance_m, max_distance_m, p90_distance_m,
                    gini_coefficient, bottom_decile_avg_distance_m
          spatial_breakdown: worst_access_points (top 10), uncovered count
          density:  facilities_per_km2, facilities_per_1000_people,
                    area_km2, n_facilities, n_demand_points
          facility_dataset_key, distance_metric_used, warnings, error.
    """
    try:
        from utils.activity_log import log_event as _log_event
    except Exception:  # pragma: no cover
        _log_event = None

    def _log(status: str, detail: str) -> None:
        if _log_event is None:
            return
        try:
            _log_event("analysis.facilities", status, detail)
        except Exception:
            pass

    _log("try", "Analysing existing facilities (coverage + access)")

    data_store = get_data()
    if not data_store:
        _log("fail", "No data loaded")
        return {"error": "No datasets loaded. Fetch data first."}

    facility_key, err = _find_facility_key(data_store, facility_dataset_key)
    if err is not None:
        _log("fail", err)
        return {"error": err}

    facilities_gdf = data_store[facility_key]
    demand_gdf = _find_demand_gdf(data_store)
    if demand_gdf is None or len(demand_gdf) == 0:
        msg = "No demand dataset loaded. Fetch population for the AOI first."
        _log("fail", msg)
        return {"error": msg}

    boundary_gdf = _find_boundary_gdf(data_store)

    unit_key = (service_radius_unit or "m").lower().strip().replace(" ", "")
    factor = _UNIT_TO_METRES.get(unit_key)
    if factor is None:
        _log("fail", f"Unknown unit '{service_radius_unit}'")
        return {"error": f"Unknown service_radius_unit '{service_radius_unit}'."}
    radius_m = float(service_radius) * factor

    network_graph: Any = None
    metric = (distance_metric or "network").lower().strip()
    if metric == "network":
        nm = get_network_manager()
        if nm is not None:
            try:
                from shapely.ops import unary_union
                boundary_polygon = None
                if boundary_gdf is not None and len(boundary_gdf) > 0:
                    try:
                        boundary_polygon = unary_union(boundary_gdf.geometry)
                    except Exception:
                        boundary_polygon = None
                network_graph = nm.get_graph(demand_gdf, boundary_polygon)
            except Exception as exc:
                logger.warning("analysis: network graph fetch failed (%s)", exc)
                network_graph = None

    try:
        from utils.facility_analysis import analyze_facilities
        result = analyze_facilities(
            demand_gdf=demand_gdf,
            facilities_gdf=facilities_gdf,
            boundary_gdf=boundary_gdf,
            service_radius_m=radius_m,
            distance_metric=metric,
            network_graph=network_graph,
        )
    except ValueError as exc:
        _log("fail", str(exc))
        return {"error": str(exc)}
    except Exception as exc:
        logger.error("analyze_existing_facilities: %s", exc, exc_info=True)
        _log("fail", f"analysis failed: {exc}")
        return {"error": f"Analysis failed: {exc}"}

    ps = get_problem_state()
    if ps is not None:
        ps["facility_analysis"] = {
            "facility_dataset_key": facility_key,
            "service_radius_m": radius_m,
            "per_point_distance_m": result.get("per_point_distance_m", []),
            "uncovered_demand_indices": result.get("uncovered_demand_indices", []),
            "summary": {
                k: v for k, v in result.items()
                if k not in ("per_point_distance_m", "uncovered_demand_indices")
            },
        }

    cov = result["coverage"]["pct_demand_covered"]
    n_unc = len(result["uncovered_demand_indices"])
    _log(
        "ok",
        f"{facility_key}: {cov:.1f}% demand covered within "
        f"{radius_m:.0f}m, {n_unc} uncovered demand points",
    )

    return {
        "facility_dataset_key": facility_key,
        "coverage": result["coverage"],
        "access": result["access"],
        "spatial_breakdown": {
            "worst_access_points": result["spatial_breakdown"]["worst_access_points"],
            "n_uncovered_demand_points": n_unc,
        },
        "density": result["density"],
        "distance_metric_used": result["distance_metric_used"],
        "warnings": result["warnings"],
        "error": None,
    }
