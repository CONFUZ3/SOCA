"""Equity metrics for facility-location solutions.

Computed post-solve and attached to every solver result so the agent's
summary always pairs the primary objective (efficiency) with a view of the
distributional impact (equity).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _gini(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Weighted Gini coefficient for a non-negative value distribution.

    Returns 0.0 for fewer than two non-zero values or any failure mode.
    """
    if values is None or len(values) < 2:
        return 0.0
    v = np.asarray(values, dtype=float)
    if np.any(np.isnan(v)) or np.all(v == 0):
        return 0.0
    if weights is None:
        weights = np.ones_like(v)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    cum_vw = np.cumsum(v_sorted * w_sorted)
    total_w = cum_w[-1]
    total_vw = cum_vw[-1]
    if total_w <= 0 or total_vw <= 0:
        return 0.0
    # Standard weighted Gini: 1 - 2 * (area under Lorenz curve).
    # The Lorenz curve must start at (0,0) — prepend origin so the
    # trapezoidal integral covers the full [0,1] population share range.
    # Omitting the origin systematically underestimates the Lorenz area,
    # which overestimates the Gini coefficient.
    lorenz = np.concatenate([[0.0], cum_vw / total_vw])
    pop_share = np.concatenate([[0.0], cum_w / total_w])
    # NumPy 2.x removed np.trapz; prefer trapezoid when available.
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area = _trapz(lorenz, pop_share)
    return float(max(0.0, min(1.0, 1.0 - 2.0 * area)))


def compute_equity_metrics(
    distance_matrix: np.ndarray,
    selected_facility_indices: list,
    assignments: Optional[Dict[int, int]],
    demand_weights: Optional[np.ndarray] = None,
    coverage_radius: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute equity-focused metrics from a solver's output.

    Args:
        distance_matrix: ``(n_demand, n_candidates)`` array, same metric the
            solver used (metres or seconds — the units flow through to the
            output transparently).
        selected_facility_indices: candidate indices the solver picked.
        assignments: optional mapping ``{demand_idx: candidate_idx}``. When
            absent (e.g. coverage models), the assigned distance is the
            min distance to any selected facility.
        demand_weights: per-demand weight (population). Defaults to all-ones.
        coverage_radius: when set, used for ``pct_demand_within_threshold``.

    Returns:
        ``{"max_weighted_distance", "gini_coefficient",
           "pct_demand_within_threshold", "bottom_decile_avg_distance"}``.
        Values that cannot be computed default to 0.0 / None.

    Failure modes: any unexpected shape returns the all-zero block with a
    WARNING log — never raises.
    """
    out: Dict[str, Any] = {
        "max_weighted_distance": 0.0,
        "gini_coefficient": 0.0,
        "pct_demand_within_threshold": None,
        "bottom_decile_avg_distance": 0.0,
    }

    try:
        if distance_matrix is None or len(selected_facility_indices) == 0:
            return out
        D = np.asarray(distance_matrix, dtype=float)
        if D.ndim != 2 or D.shape[0] == 0 or D.shape[1] == 0:
            return out
        sel = [int(i) for i in selected_facility_indices if 0 <= int(i) < D.shape[1]]
        if not sel:
            return out

        n_demand = D.shape[0]
        if demand_weights is None or len(demand_weights) != n_demand:
            w = np.ones(n_demand, dtype=float)
        else:
            w = np.asarray(demand_weights, dtype=float)
            w = np.where(np.isnan(w), 0.0, w)

        # Per-demand assigned distance.
        if assignments:
            assigned = np.full(n_demand, np.nan, dtype=float)
            for d_idx, f_idx in assignments.items():
                try:
                    di = int(d_idx); fi = int(f_idx)
                    if 0 <= di < n_demand and 0 <= fi < D.shape[1]:
                        assigned[di] = D[di, fi]
                except Exception:
                    continue
            # Fill any unassigned with min over selected (coverage fallback).
            mask = np.isnan(assigned)
            if mask.any():
                assigned[mask] = D[mask][:, sel].min(axis=1)
        else:
            assigned = D[:, sel].min(axis=1)

        out["max_weighted_distance"] = float(np.max(w * assigned)) if n_demand else 0.0
        out["gini_coefficient"] = _gini(assigned, w)

        if coverage_radius is not None and coverage_radius > 0:
            within = assigned <= float(coverage_radius)
            total_w = float(w.sum())
            out["pct_demand_within_threshold"] = (
                float(w[within].sum() / total_w * 100.0) if total_w > 0 else 0.0
            )

        # Bottom decile by worst-served (largest distances), weighted.
        order = np.argsort(-assigned)
        cum_w = np.cumsum(w[order])
        total_w = float(w.sum())
        if total_w > 0:
            cutoff = 0.10 * total_w
            take = order[cum_w <= cutoff]
            if len(take) == 0:
                take = order[:1]  # at least one point
            out["bottom_decile_avg_distance"] = float(
                np.average(assigned[take], weights=w[take])
                if w[take].sum() > 0 else assigned[take].mean()
            )
    except Exception as exc:
        logger.warning("compute_equity_metrics: failed (%s); returning zeros", exc)

    return out
