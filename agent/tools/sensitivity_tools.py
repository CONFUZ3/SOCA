"""ADK tool: drop-one sensitivity analysis on a completed optimization.

For each selected facility in the last solution, re-runs the same solver
with that facility excluded from the candidate set. Reports the
degradation in objective value (sign-correct for minimisation vs.
maximisation problems) and flags the facility whose removal hurts the
solution most as ``most_critical``.

Re-uses the session's cached road-network graph (re-fetching only if the
LRU cache was evicted) and current data_dict.
"""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.tools.tool_context import ToolContext

from .state_bridge import (
    get_data,
    get_problem_state,
    get_problem_registry,
)

logger = logging.getLogger(__name__)


# Problem types where the objective is *maximised* (more is better).
_MAXIMISATION_TYPES = {"mclp"}


def run_sensitivity_analysis(
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Drop-one re-optimization on the most recent solution.

    For each facility selected by the last solve, re-runs the solver with
    that facility dropped from the candidate pool (same p), and records:
      - ``objective_without``: objective value of the new solve
      - ``objective_degradation_pct``: percentage degradation (positive =
        worse). For minimisation problems (P-Median, P-Center, LSCP),
        ``+%`` means the objective grew. For maximisation (MCLP), ``+%``
        means coverage shrank.
      - ``newly_uncovered_demand_weight``: only meaningful for coverage
        models; weight of demand that the original solution covered but
        the perturbed one does not.

    Design note — p vs p-1: this analysis re-solves with the same p from
    n-1 candidates, which answers "how replaceable is this facility?" (the
    solver substitutes the best available alternative). The alternative
    convention — re-solve with p-1 — answers "what is the absolute value
    of having this facility?". The replaceability framing is more useful
    for practitioners deciding which facilities to protect or harden, and
    avoids conflating facility criticality with the marginal value of p.

    Returns:
        ``{"sensitivity": [...], "most_critical_facility_id": int|None,
           "ran_n_reoptimizations": int, "error": str|None}``.

    Failure modes: returns ``{"error": ...}`` if there is no completed
    solution, the registry is unavailable, or the candidate set is too
    small to drop a facility.
    """
    if tool_context is None:
        return {"error": "No tool context available."}

    ps = get_problem_state()
    last = ps.get("solution") if ps else None
    if not last or not isinstance(last, dict):
        return {"error": "No completed optimization found. Run confirm_optimization first."}
    selected = list(last.get("selected_facilities") or [])
    if not selected:
        return {"error": "Last solution has no selected facilities to analyse."}

    base_obj = last.get("objective_value")
    if base_obj is None:
        return {"error": "Last solution has no objective_value."}

    problem_type = last.get("problem_type") or ps.get("problem_type")
    if not problem_type:
        return {"error": "Cannot determine problem_type for the last solution."}

    registry = get_problem_registry()
    if not registry:
        return {"error": "Problem registry not available."}
    problem_solver = registry.get_problem(problem_type)
    if not problem_solver:
        return {"error": f"Problem type '{problem_type}' not found in registry."}

    parameters = dict(ps.get("parameters") or {})
    distance_metric = last.get("distance_metric_used", "network")

    # Re-build data_dict using same _prepare_solver_inputs logic so we
    # operate on the exact data the original solve used.
    from agent.tools.optimize_tools import (
        _prepare_solver_inputs,
        _categorise_data,
    )

    data_store = get_data()
    boundary_keys, demand_keys, candidate_keys = _categorise_data(data_store)
    data_dict, parameters, err = _prepare_solver_inputs(
        data_store, boundary_keys, demand_keys, candidate_keys,
        problem_type, problem_solver, parameters,
    )
    if err is not None:
        return {"error": err.get("error_message", "Failed to assemble data.")}

    cand_gdf = data_dict.get("candidate_sites")
    if cand_gdf is None or len(cand_gdf) <= 1:
        return {"error": "Candidate set has fewer than 2 sites — cannot drop one."}

    # Re-acquire the road-network graph the baseline used. It lives in the
    # session NetworkManager's LRU cache (not in problem_state), so we fetch
    # it via the same helper confirm_optimization uses — a cache hit for the
    # unchanged AOI. Without this the solvers silently downgrade to geodesic,
    # inflating absolute coverage/objective relative to the network baseline.
    network_graph = None
    network_warnings: list = []
    if distance_metric == "network":
        from agent.tools.optimize_tools import _fetch_network_graph
        network_graph, distance_metric, network_warnings, _ = _fetch_network_graph(
            distance_metric, False, data_dict, boundary_keys, data_store,
        )

    is_max = problem_type in _MAXIMISATION_TYPES

    results = []
    for fac_idx in selected:
        try:
            i = int(fac_idx)
        except (TypeError, ValueError):
            continue
        if i < 0 or i >= len(cand_gdf):
            continue
        try:
            reduced = cand_gdf.drop(cand_gdf.index[i]).reset_index(drop=True)
            sub_data = dict(data_dict)
            sub_data["candidate_sites"] = reduced
            # Solvers read the road graph from data['_network_graph'] (see
            # optimize_tools confirm path); inject the re-acquired graph so the
            # re-solve uses the same network distances as the baseline.
            if network_graph is not None:
                sub_data["_network_graph"] = network_graph
            try:
                sub_solution = problem_solver.solve(
                    data=sub_data,
                    parameters=parameters,
                    constraints={},
                    distance_metric=distance_metric,
                )
            except Exception as exc:
                results.append({
                    "facility_index": i,
                    "objective_without": None,
                    "objective_degradation_pct": None,
                    "error": f"solve failed: {exc}",
                })
                continue
            obj_without = sub_solution.get("objective_value")
            if obj_without is None or base_obj == 0:
                deg = None
            else:
                # Sign-correct: positive % = worse than baseline.
                if is_max:
                    deg = (base_obj - obj_without) / abs(base_obj) * 100.0
                else:
                    deg = (obj_without - base_obj) / abs(base_obj) * 100.0
            results.append({
                "facility_index": i,
                "objective_without": obj_without,
                "objective_degradation_pct": deg,
                "newly_uncovered_demand_weight": _coverage_loss(
                    last, sub_solution, data_dict
                ) if is_max else None,
            })
        except Exception as exc:
            logger.warning("sensitivity: drop-one for facility %s failed (%s)", i, exc)
            results.append({
                "facility_index": i,
                "objective_without": None,
                "objective_degradation_pct": None,
                "error": str(exc),
            })

    # Sort descending by degradation; missing values sort last.
    def _key(r):
        v = r.get("objective_degradation_pct")
        return (-1e18) if v is None else v
    sorted_results = sorted(results, key=_key, reverse=True)
    most_critical = None
    if sorted_results and sorted_results[0].get("objective_degradation_pct") is not None:
        most_critical = sorted_results[0]["facility_index"]
        sorted_results[0]["most_critical"] = True

    return {
        "sensitivity": sorted_results,
        "most_critical_facility_id": most_critical,
        "ran_n_reoptimizations": len(sorted_results),
        "problem_type": problem_type,
        "baseline_objective_value": base_obj,
        "distance_metric_used": distance_metric,
        "warnings": network_warnings,
        "error": None,
    }


def _coverage_loss(base_solution: dict, perturbed_solution: dict, data_dict: dict) -> Optional[float]:
    """Approximate demand weight lost when switching from baseline to perturbed.

    Computed as the sum of weights for demand assigned in the baseline but
    not in the perturbed solution. Returns None if either solution lacks
    assignment data.
    """
    base_assign = base_solution.get("assignments") or {}
    pert_assign = perturbed_solution.get("assignments") or {}
    if not base_assign or not pert_assign:
        return None
    try:
        import pandas as _pd
        demand = data_dict.get("demand_points")
        weights = None
        if demand is not None:
            for col in ("default_weight", "weight", "population", "demand"):
                if col in demand.columns:
                    weights = _pd.to_numeric(demand[col], errors="coerce").fillna(1.0).values
                    break
        lost = 0.0
        for d_idx in base_assign:
            if d_idx not in pert_assign:
                w = float(weights[int(d_idx)]) if weights is not None and int(d_idx) < len(weights) else 1.0
                lost += w
        return lost
    except Exception:
        return None
