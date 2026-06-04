"""Rich, fully unit-labeled facts block for post-solve analysis.

The agent (LLM) now writes the narrative analysis itself; this module's job is
to hand it a complete, trustworthy set of *facts* — every number labeled with
its unit, facilities named by place, coverage gaps surfaced with locations, and
equity expressed as interpretable ratios. See ``build_analysis_facts``.

All distances are reported in **kilometres**. The underlying distance matrix is
in metres (``utils.distance_calculator``), so conversions happen here once.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Don't reverse-geocode huge facility sets (latency / external-API courtesy).
_MAX_FACILITIES_TO_LABEL = 20
# Worst-served / uncovered points to surface in coverage_gaps.
_MAX_GAP_POINTS = 8

_METRIC_LABELS = {
    "network": "road-network shortest path (OSM)",
    "geodesic": "geodesic great-circle distance",
    "haversine": "geodesic great-circle distance",
    "euclidean": "straight-line (Euclidean) distance",
}


def _reverse_geocode_enabled() -> bool:
    return os.environ.get("SOCA_REVERSE_GEOCODE", "1") not in ("0", "false", "False")


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Population-weighted percentile (q in [0, 1])."""
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    total = cum[-1]
    if total <= 0:
        return float(v[-1])
    cutoff = q * total
    idx = int(np.searchsorted(cum, cutoff))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def _coords_lonlat(gdf) -> Optional[np.ndarray]:
    """Return an (n, 2) array of [lon, lat], reprojecting to EPSG:4326 if needed."""
    try:
        g = gdf
        if getattr(g, "crs", None) is not None and str(g.crs).upper() not in (
            "EPSG:4326", "WGS84", "EPSG:4326",
        ):
            try:
                g = g.to_crs(4326)
            except Exception:
                g = gdf
        pts = g.geometry.representative_point()
        return np.column_stack([pts.x.to_numpy(), pts.y.to_numpy()])
    except Exception as exc:
        logger.debug("solution_report: coord extraction failed (%s)", exc)
        return None


def _assigned_distances_m(
    D: np.ndarray, sel: List[int], assignments: Optional[Dict[int, int]]
) -> np.ndarray:
    """Per-demand assigned distance in metres (assignment, else nearest selected)."""
    n = D.shape[0]
    if assignments:
        out = np.full(n, np.nan, dtype=float)
        for d_idx, f_idx in assignments.items():
            try:
                di, fi = int(d_idx), int(f_idx)
                if 0 <= di < n and 0 <= fi < D.shape[1]:
                    out[di] = D[di, fi]
            except Exception:
                continue
        mask = np.isnan(out)
        if mask.any() and sel:
            out[mask] = D[mask][:, sel].min(axis=1)
        return out
    if sel:
        return D[:, sel].min(axis=1)
    return np.full(n, np.nan, dtype=float)


def _nearest_selected(D: np.ndarray, sel: List[int]) -> np.ndarray:
    """For each demand point, the selected facility index nearest to it."""
    sub = D[:, sel]
    return np.asarray([sel[j] for j in sub.argmin(axis=1)])


def build_analysis_facts(
    solution: dict,
    data_dict: dict,
    parameters: dict,
    distance_metric: str,
    distance_matrix: Optional[np.ndarray] = None,
    demand_weights: Optional[np.ndarray] = None,
    equity: Optional[dict] = None,
) -> Optional[dict]:
    """Build the structured facts block the agent narrates from.

    Returns ``None`` (caller falls back to the legacy summary) only if the
    essential inputs are missing. Otherwise always returns a dict; individual
    sections degrade gracefully to ``None``/empty on error.
    """
    try:
        demand_gdf = data_dict.get("demand_points")
        cand_gdf = data_dict.get("candidate_sites")
        D = distance_matrix
        if demand_gdf is None or cand_gdf is None or D is None:
            return None
        D = np.asarray(D, dtype=float)
        if D.ndim != 2 or D.shape[0] == 0:
            return None

        sel = [int(i) for i in (solution.get("selected_facilities") or [])
               if 0 <= int(i) < D.shape[1]]
        if not sel:
            return None

        assignments = solution.get("assignments") or {}
        n_demand = D.shape[0]

        if demand_weights is None or len(demand_weights) != n_demand:
            w = np.ones(n_demand, dtype=float)
        else:
            w = np.nan_to_num(np.asarray(demand_weights, dtype=float), nan=0.0)
        total_w = float(w.sum()) or 1.0

        assigned_m = _assigned_distances_m(D, sel, assignments)
        valid = ~np.isnan(assigned_m)
        assigned_km = assigned_m[valid] / 1000.0
        wv = w[valid]
        wkm_total = float(wv.sum()) or 1.0

        # ---- units -------------------------------------------------------
        metric_key = (distance_metric or "").lower()
        units = {
            "distance": "km",
            "distance_metric": metric_key or "unknown",
            "distance_metric_label": _METRIC_LABELS.get(metric_key, metric_key or "unknown"),
        }

        # ---- distance distribution (population-weighted) -----------------
        mean_km = float(np.average(assigned_km, weights=wv)) if len(assigned_km) else 0.0
        distribution = {
            "mean_km": round(mean_km, 3),
            "median_km": round(_weighted_percentile(assigned_km, wv, 0.50), 3),
            "p90_km": round(_weighted_percentile(assigned_km, wv, 0.90), 3),
            "p95_km": round(_weighted_percentile(assigned_km, wv, 0.95), 3),
            "max_km": round(float(assigned_km.max()), 3) if len(assigned_km) else 0.0,
            "min_km": round(float(assigned_km.min()), 3) if len(assigned_km) else 0.0,
            "std_km": round(float(assigned_km.std()), 3) if len(assigned_km) else 0.0,
            "weighting": "population-weighted",
        }

        # ---- service radius / coverage -----------------------------------
        radius_m = _service_radius_m(parameters)
        coverage: Optional[dict] = None
        if radius_m is not None:
            covered_mask = assigned_m <= radius_m
            covered_w = float(w[valid & covered_mask].sum())
            coverage = {
                "service_radius_km": round(radius_m / 1000.0, 3),
                "pct_demand_covered": round(covered_w / total_w * 100.0, 2),
                "covered_demand_weight": round(covered_w, 2),
                "uncovered_demand_weight": round(total_w - covered_w, 2),
                "num_uncovered_points": int(np.sum(valid & ~covered_mask)),
            }

        # ---- per-facility breakdown (named) ------------------------------
        facilities = _facility_breakdown(
            D, sel, assignments, assigned_m, w, cand_gdf, metric_key
        )

        # ---- coverage gaps (worst-served / uncovered, located) -----------
        gaps = _coverage_gaps(assigned_m, w, demand_gdf, radius_m)

        # ---- equity (interpretable) --------------------------------------
        equity_block = _equity_block(equity, assigned_km, wv, mean_km)

        # ---- solver technicals -------------------------------------------
        sd = solution.get("solver_details") or {}
        solver_block = {
            "solver": solution.get("solver") or sd.get("solver") or "unknown",
            "mip_gap": sd.get("gap"),
            "solve_time_seconds": round(float(solution.get("solver_time_seconds", 0) or 0), 2),
            "timed_out": bool(sd.get("timed_out", False)),
            "formulation": sd.get("formulation"),
            "status": solution.get("status"),
        }

        return {
            "problem_type": solution.get("problem_type"),
            "variant": solution.get("variant", "base"),
            "objective": {
                "name": (solution.get("metrics") or {}).get("objective_name"),
                "value": solution.get("objective_value"),
            },
            "scope": {
                "num_demand_points": int(n_demand),
                "num_candidate_sites": int(D.shape[1]),
                "num_facilities_selected": len(sel),
                "total_demand_weight": round(total_w, 2),
            },
            "units": units,
            "distance_distribution": distribution,
            "coverage": coverage,
            "facilities": facilities,
            "coverage_gaps": gaps,
            "equity": equity_block,
            "solver": solver_block,
            "warnings": solution.get("warnings") or [],
        }
    except Exception as exc:
        logger.warning("build_analysis_facts failed (%s)", exc)
        return None


def _service_radius_m(parameters: dict) -> Optional[float]:
    sr = parameters.get("service_radius")
    if sr is None:
        return None
    try:
        from utils.distance_calculator import DistanceCalculator
        return float(DistanceCalculator()._convert_to_meters(
            float(sr), parameters.get("service_radius_unit")
        ))
    except Exception:
        try:
            return float(sr)
        except (TypeError, ValueError):
            return None


def _facility_breakdown(
    D, sel, assignments, assigned_m, w, cand_gdf, metric_key
) -> List[dict]:
    coords = _coords_lonlat(cand_gdf)
    # Which selected facility serves each demand point.
    if assignments:
        served_by = np.full(D.shape[0], -1, dtype=int)
        for d_idx, f_idx in assignments.items():
            try:
                served_by[int(d_idx)] = int(f_idx)
            except Exception:
                continue
        unset = served_by < 0
        if unset.any():
            served_by[unset] = _nearest_selected(D[unset], sel)
    else:
        served_by = _nearest_selected(D, sel)

    label_them = _reverse_geocode_enabled() and len(sel) <= _MAX_FACILITIES_TO_LABEL
    out: List[dict] = []
    for f in sel:
        mask = served_by == f
        dists_m = assigned_m[mask]
        dists_m = dists_m[~np.isnan(dists_m)]
        wm = w[mask]
        rec: dict = {
            "index": int(f),
            "num_demand_points": int(np.sum(mask)),
            "demand_served_weight": round(float(wm.sum()), 2),
            "avg_distance_km": round(
                float(np.average(dists_m, weights=wm[: len(dists_m)])) / 1000.0, 3
            ) if len(dists_m) and wm[: len(dists_m)].sum() > 0 else None,
            "max_distance_km": round(float(dists_m.max()) / 1000.0, 3) if len(dists_m) else None,
        }
        if coords is not None and f < len(coords):
            lon, lat = float(coords[f][0]), float(coords[f][1])
            rec["lat"], rec["lon"] = round(lat, 6), round(lon, 6)
            if label_them:
                try:
                    from utils.geocoder import reverse_geocode
                    rec["place"] = reverse_geocode(lat, lon)
                except Exception:
                    rec["place"] = None
        out.append(rec)
    out.sort(key=lambda r: r.get("demand_served_weight", 0), reverse=True)
    return out


def _coverage_gaps(assigned_m, w, demand_gdf, radius_m) -> List[dict]:
    coords = _coords_lonlat(demand_gdf)
    valid = ~np.isnan(assigned_m)
    idxs = np.where(valid)[0]
    if len(idxs) == 0:
        return []
    if radius_m is not None:
        # Uncovered points only, worst first.
        uncovered = idxs[assigned_m[idxs] > radius_m]
        cand = uncovered[np.argsort(-assigned_m[uncovered])]
    else:
        cand = idxs[np.argsort(-assigned_m[idxs])]
    cand = cand[:_MAX_GAP_POINTS]
    gaps: List[dict] = []
    for i in cand:
        rec = {
            "demand_index": int(i),
            "distance_km": round(float(assigned_m[i]) / 1000.0, 3),
            "demand_weight": round(float(w[i]), 2),
        }
        if coords is not None and i < len(coords):
            rec["lat"], rec["lon"] = round(float(coords[i][1]), 6), round(float(coords[i][0]), 6)
        gaps.append(rec)
    return gaps


def _equity_block(equity, assigned_km, wv, mean_km) -> dict:
    eq = dict(equity or {})
    gini = eq.get("gini_coefficient", 0.0)
    bottom_decile_km = round(float(eq.get("bottom_decile_avg_distance", 0.0)) / 1000.0, 3)
    ratio = round(bottom_decile_km / mean_km, 2) if mean_km > 0 else None
    return {
        "gini_coefficient": round(float(gini), 3),
        "mean_distance_km": round(mean_km, 3),
        "bottom_decile_avg_distance_km": bottom_decile_km,
        "bottom_decile_vs_mean_ratio": ratio,
        "pct_demand_within_threshold": eq.get("pct_demand_within_threshold"),
        "worst_case_distance_km": round(float(assigned_km.max()), 3) if len(assigned_km) else 0.0,
    }
