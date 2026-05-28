"""Read-only diagnostic analysis of already-fetched facilities.

Computes coverage, access, spatial breakdown, and density metrics for a
fixed set of existing facilities (no optimization). Reuses
``DistanceCalculator`` and ``compute_equity_metrics``; performs no new
fetching or solving.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from config.settings import settings
from utils.distance_calculator import DistanceCalculator
from utils.equity_metrics import compute_equity_metrics

logger = logging.getLogger(__name__)


_WEIGHT_COLS = ("default_weight", "weight", "population", "pop", "demand")


def _extract_weights(demand_gdf: gpd.GeoDataFrame) -> np.ndarray:
    for col in _WEIGHT_COLS:
        if col in demand_gdf.columns:
            try:
                return (
                    pd.to_numeric(demand_gdf[col], errors="coerce")
                    .fillna(1.0)
                    .to_numpy()
                )
            except Exception:
                continue
    return np.ones(len(demand_gdf), dtype=float)


def _boundary_area_km2(boundary_gdf: Optional[gpd.GeoDataFrame]) -> float:
    if boundary_gdf is None or len(boundary_gdf) == 0:
        return 0.0
    try:
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        total = 0.0
        for geom in boundary_gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            area_m2, _ = geod.geometry_area_perimeter(geom)
            total += abs(area_m2)
        return total / 1_000_000.0
    except Exception as exc:
        logger.warning("facility_analysis: geodesic area failed (%s)", exc)
        try:
            projected = boundary_gdf.to_crs(getattr(settings, "CRS_PROJECTED", "EPSG:3857"))
            return float(projected.area.sum()) / 1_000_000.0
        except Exception:
            return 0.0


def analyze_facilities(
    demand_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    boundary_gdf: Optional[gpd.GeoDataFrame],
    service_radius_m: float,
    distance_metric: str = "network",
    network_graph: Optional[Any] = None,
    top_n_worst: int = 10,
) -> Dict[str, Any]:
    """Compute coverage, access, spatial breakdown, and density metrics.

    Returns a dict with ``coverage``, ``access``, ``spatial_breakdown``,
    ``density``, ``warnings``, and the per-point distances and uncovered
    index list so callers can drive map layers.
    """
    warnings: List[str] = []

    if demand_gdf is None or len(demand_gdf) == 0:
        raise ValueError("demand layer is empty — cannot analyse facilities")
    if facilities_gdf is None or len(facilities_gdf) == 0:
        raise ValueError("facility layer is empty — nothing to analyse")
    if service_radius_m is None or float(service_radius_m) <= 0:
        raise ValueError("service_radius_m must be > 0")

    radius_m = float(service_radius_m)

    dist_calc = DistanceCalculator()
    metric = distance_metric or "network"
    if metric == "network" and network_graph is None:
        metric = "euclidean"
        warnings.append(
            "Road-network graph not available; using geodesic distance for this analysis."
        )

    # Bound the Dijkstra search radius for network distance. Points farther
    # than ``cutoff_m`` from any facility are by definition outside any
    # reasonable coverage envelope; their cells stay inf and get filled by
    # the geodesic fallback inside _network_distance, which is a fine lower
    # bound for "very far". Only matters when metric == "network".
    cutoff_m = max(radius_m * 5.0, 5000.0) if metric == "network" else None

    try:
        D = dist_calc.calculate_distance_matrix(
            demand_gdf,
            facilities_gdf,
            metric=metric,
            network_graph=network_graph,
            cutoff_m=cutoff_m,
        )
    except Exception as exc:
        logger.warning(
            "facility_analysis: %s distance failed (%s); falling back to euclidean",
            metric, exc,
        )
        warnings.append(
            f"{metric} distance failed ({exc}); used geodesic fallback."
        )
        D = dist_calc.calculate_distance_matrix(
            demand_gdf, facilities_gdf, metric="euclidean"
        )

    weights = _extract_weights(demand_gdf)
    total_w = float(weights.sum()) if len(weights) else 0.0

    nearest = D.min(axis=1)
    covered_mask = nearest <= radius_m

    covered_weight = float(weights[covered_mask].sum()) if total_w > 0 else 0.0
    pct_demand_covered = (
        (covered_weight / total_w * 100.0) if total_w > 0 else 0.0
    )
    pct_points_covered = (
        float(covered_mask.sum()) / len(covered_mask) * 100.0
        if len(covered_mask) else 0.0
    )

    if total_w > 0:
        avg_distance = float(np.average(nearest, weights=weights))
    else:
        avg_distance = float(nearest.mean()) if len(nearest) else 0.0
    max_distance = float(nearest.max()) if len(nearest) else 0.0
    p90_distance = float(np.percentile(nearest, 90)) if len(nearest) else 0.0

    n_facilities = len(facilities_gdf)
    equity = compute_equity_metrics(
        D,
        list(range(n_facilities)),
        assignments=None,
        demand_weights=weights,
        coverage_radius=radius_m,
    )

    uncovered_indices: List[int] = [int(i) for i in np.where(~covered_mask)[0].tolist()]

    order_worst = np.argsort(-nearest)
    worst_take = order_worst[:max(0, int(top_n_worst))]
    try:
        d4326 = demand_gdf.to_crs("EPSG:4326") if demand_gdf.crs is not None else demand_gdf
    except Exception:
        d4326 = demand_gdf
    worst_points = []
    for idx in worst_take:
        try:
            geom = d4326.geometry.iloc[int(idx)]
            worst_points.append({
                "demand_idx": int(idx),
                "distance_m": float(nearest[int(idx)]),
                "weight": float(weights[int(idx)]) if int(idx) < len(weights) else 1.0,
                "lat": float(geom.y),
                "lon": float(geom.x),
            })
        except Exception:
            continue

    area_km2 = _boundary_area_km2(boundary_gdf)
    facilities_per_km2 = (n_facilities / area_km2) if area_km2 > 0 else 0.0
    facilities_per_1000_people = (
        (n_facilities / total_w * 1000.0) if total_w > 0 else 0.0
    )

    return {
        "coverage": {
            "pct_demand_covered": pct_demand_covered,
            "pct_points_covered": pct_points_covered,
            "uncovered_demand_weight": max(0.0, total_w - covered_weight),
            "service_radius_m": radius_m,
        },
        "access": {
            "avg_distance_m": avg_distance,
            "max_distance_m": max_distance,
            "p90_distance_m": p90_distance,
            "gini_coefficient": equity.get("gini_coefficient", 0.0),
            "bottom_decile_avg_distance_m": equity.get(
                "bottom_decile_avg_distance", 0.0
            ),
        },
        "spatial_breakdown": {
            "worst_access_points": worst_points,
            "uncovered_demand_indices": uncovered_indices,
        },
        "density": {
            "facilities_per_km2": facilities_per_km2,
            "facilities_per_1000_people": facilities_per_1000_people,
            "area_km2": area_km2,
            "n_facilities": n_facilities,
            "n_demand_points": int(len(demand_gdf)),
        },
        "distance_metric_used": metric,
        "per_point_distance_m": [float(v) for v in nearest.tolist()],
        "uncovered_demand_indices": uncovered_indices,
        "warnings": warnings,
    }
