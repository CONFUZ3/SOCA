"""
ADK tools for spatial optimization.

stage_optimization   – Validates parameters, builds a preview, and stores them
                       in ADK session state awaiting user confirmation.
confirm_optimization – Reads staged parameters and runs the solver.  Writes
                       the solution directly into the Streamlit problem_state.
"""

import logging
import time
import inspect
from typing import Optional

from google.adk.tools.tool_context import ToolContext

from .state_bridge import (
    get_data,
    get_problem_state,
    get_problem_registry,
    get_generated_sites_count,
    get_generated_sites_seed,
    get_network_manager,
)

logger = logging.getLogger(__name__)

# Unit conversion factors to metres
_UNIT_TO_METRES = {
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "yd": 0.9144,
    "yard": 0.9144,
    "yards": 0.9144,
    "nm": 1852.0,
    "nmi": 1852.0,
}

_VALID_PROBLEM_TYPES = ("p-median", "p-center", "mclp", "lscp")


def _convert_radius(value: float, unit: str) -> tuple:
    """Return (value_in_metres, normalised_unit)."""
    key = unit.lower().strip().replace(" ", "")
    factor = _UNIT_TO_METRES.get(key)
    if factor is None:
        logger.warning("Unknown radius unit '%s', treating as metres", unit)
        return value, "m"
    normalised = key if key in ("km", "m", "miles", "ft", "yd", "nm", "nmi") else "m"
    return value * factor, normalised


def _categorise_data(data_store: dict) -> tuple:
    """Split data_store into (boundary_keys, demand_keys, candidate_keys)."""
    from utils.data_processor import DataProcessor
    processor = DataProcessor()

    boundary_keys: set = set()
    candidate_keys: set = set()
    demand_keys: set = set()

    for name, gdf in data_store.items():
        src = gdf.attrs.get("source", "")
        key_lower = name.lower()

        # Boundary detection
        if key_lower.startswith("boundary_") or src in (
            "auto_fetched", "osmnx", "photon_bbox_fallback",
            "nominatim", "nominatim_bbox_fallback", "gadm",
        ):
            if len(gdf) > 0 and gdf.geometry.iloc[0].geom_type in (
                "Polygon", "MultiPolygon"
            ):
                boundary_keys.add(name)
                continue

        # POI / facility datasets → candidates
        if "_facilities_" in key_lower or any(
            key_lower.startswith(cat + "_")
            for cat in ["health", "education", "food", "finance",
                        "fire_station", "police", "library"]
        ):
            candidate_keys.add(name)
            continue

        if key_lower == "generated_candidates":
            candidate_keys.add(name)
            continue

        # Column/name inference
        data_type = processor.identify_data_type(gdf)
        if data_type == "demand_points" or "demand" in key_lower:
            demand_keys.add(name)
        elif data_type == "candidate_sites" or any(
            w in key_lower for w in ["candidate", "site", "facility"]
        ):
            candidate_keys.add(name)
        else:
            demand_keys.add(name)

    return boundary_keys, demand_keys, candidate_keys


def stage_optimization(
    problem_type: str,
    n_facilities: Optional[int] = None,
    service_radius: Optional[float] = None,
    service_radius_unit: str = "m",
    variant: str = "base",
    budget: Optional[float] = None,
    k_coverage: Optional[int] = None,
    max_assignment_distance: Optional[float] = None,
    facility_reliability: Optional[float] = None,
    demand_weight_column: Optional[str] = None,
    objective: str = "total",
    distance_metric: str = "network",
    strict_network: bool = False,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Stage an optimization run for user confirmation.

    Validates the problem type and parameters, shows the user what will be
    optimized, and stores the parameters in session state.  Does NOT run the
    solver – call confirm_optimization() after the user says "yes".

    Args:
        problem_type: One of "p-median", "p-center", "mclp", "lscp".
        n_facilities: Number of facilities to open (required for all except LSCP).
        service_radius: Coverage radius (required for MCLP and LSCP).
        service_radius_unit: Unit for service_radius. One of: m, km, miles, ft,
                             yd, nm.  Default is metres.
        variant: Problem variant. p-median: base / capacitated / budget /
                 max_distance.  mclp: classical / capacitated / budget /
                 multi_coverage / backup / probabilistic.
        budget: Total budget (required for budget variants).
        k_coverage: Minimum coverage redundancy (required for multi_coverage /
                    backup MCLP variants).
        max_assignment_distance: Max assignment distance in metres (required for
                                 p-median max_distance variant).
        facility_reliability: Reliability probability 0-1 (probabilistic MCLP).
        demand_weight_column: Column name for demand weights (only needed for
                              non-standard column names).
        objective: Objective type for p-median: "total" or "average".
        distance_metric: Distance calculation method. Default is "network"
                         (road-network shortest path via OpenStreetMap; the road
                         graph is pre-fetched in the background when the AOI is
                         confirmed, so most runs see a warm cache). Use
                         "euclidean" for straight-line/geodesic distance or
                         "manhattan" for grid distance. On a network-fetch
                         failure the tool auto-falls back to geodesic and
                         surfaces a warning unless strict_network=True.
        strict_network: If True, abort with an error when the road-network
                        fetch fails instead of falling back to geodesic.
                        Default False. Set True only when the user insists on
                        road-network distance for reproducibility.

    Returns:
        dict with keys:
          staged (bool): True if successfully staged.
          problem_type, parameters: what was staged.
          data_status: summary of available datasets.
          validation_warnings (list[str]): non-fatal warnings.
          error (str | None): reason if staged=False.
    """
    registry = get_problem_registry()
    data_store = get_data()

    # Validate problem type
    pt = (problem_type or "").lower().strip()
    if registry:
        problem_solver = registry.get_problem(pt)
        if not problem_solver:
            # Try fuzzy match
            inferred = registry.infer_problem_type(pt)
            if inferred:
                pt = inferred
                problem_solver = registry.get_problem(pt)

    if pt not in _VALID_PROBLEM_TYPES:
        return {
            "staged": False,
            "error": (
                f"Unknown problem type '{problem_type}'. "
                f"Valid options: {', '.join(_VALID_PROBLEM_TYPES)}"
            ),
        }

    # Check data availability
    boundary_keys, demand_keys, candidate_keys = _categorise_data(data_store)
    has_demand = bool(demand_keys)
    has_candidates = bool(candidate_keys)

    if not has_demand:
        return {
            "staged": False,
            "error": (
                "No demand data available. Please upload demand points or use "
                "fetch_city_data() to load population data automatically."
            ),
        }

    # Normalise variant
    variant = (variant or "base").lower().replace("-", "_").strip()

    # Convert service_radius to metres
    radius_metres: Optional[float] = None
    radius_unit_norm: str = service_radius_unit or "m"
    if service_radius is not None:
        radius_metres, radius_unit_norm = _convert_radius(
            service_radius, service_radius_unit or "m"
        )

    # Build parameter dict
    parameters: dict = {}
    if n_facilities is not None:
        parameters["n_facilities"] = n_facilities
    if radius_metres is not None:
        parameters["service_radius"] = radius_metres
        parameters["service_radius_original"] = service_radius
        parameters["service_radius_unit"] = radius_unit_norm
    if variant != "base":
        parameters["variant"] = variant
    if budget is not None:
        parameters["budget"] = budget
    if k_coverage is not None:
        parameters["k_coverage"] = k_coverage
    if max_assignment_distance is not None:
        parameters["max_assignment_distance"] = max_assignment_distance
    if facility_reliability is not None:
        parameters["facility_reliability"] = facility_reliability
    if demand_weight_column is not None:
        parameters["demand_weight_column"] = demand_weight_column
    if objective != "total":
        parameters["objective"] = objective
    if distance_metric != "network":
        parameters["distance_metric"] = distance_metric

    # Variant-specific validation warnings
    warnings: list = []
    if pt == "p-median":
        if variant == "budget" and ("budget" not in parameters or "facility_costs" not in parameters):
            warnings.append(
                "Budget P-Median needs 'budget' and 'facility_costs'. "
                "Please provide them or switch to 'base' variant."
            )
        if variant == "capacitated" and "capacities" not in parameters:
            warnings.append(
                "Capacitated P-Median needs 'capacities'. Will auto-detect from data."
            )
        if variant == "max_distance" and "max_assignment_distance" not in parameters:
            warnings.append(
                "Max-distance P-Median needs 'max_assignment_distance'."
            )
    elif pt == "mclp":
        if radius_metres is None:
            return {
                "staged": False,
                "error": "MCLP requires a service_radius. Please specify one (with unit, e.g. 5 km).",
            }
        if variant in ("multi_coverage", "backup") and "k_coverage" not in parameters:
            warnings.append(
                f"{variant.replace('_', ' ').title()} MCLP needs 'k_coverage'. Defaulting to 2."
            )
            parameters["k_coverage"] = 2
    elif pt == "lscp":
        if radius_metres is None:
            return {
                "staged": False,
                "error": "LSCP requires a service_radius.",
            }

    # Stage into ADK session state
    pending = {
        "problem_type": pt,
        "parameters": parameters,
        "constraints": {},
        "distance_metric": distance_metric,
        "strict_network": bool(strict_network),
    }
    if tool_context is not None:
        tool_context.state["pending_optimization"] = pending
        # Also update visible problem state fields so agent sees them
        tool_context.state["problem_type"] = pt
        tool_context.state["parameters"] = parameters

    # Also update Streamlit problem_state directly
    ps = get_problem_state()
    ps["problem_type"] = pt
    ps["parameters"] = parameters

    data_status = {
        "demand_datasets": list(demand_keys),
        "candidate_datasets": list(candidate_keys),
        "boundary_datasets": list(boundary_keys),
        "will_generate_candidates": has_demand and not has_candidates,
    }

    return {
        "staged": True,
        "problem_type": pt,
        "parameters": parameters,
        "data_status": data_status,
        "validation_warnings": warnings,
        "error": None,
    }


def confirm_optimization(
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Run the staged optimization after user confirmation.

    Call this ONLY after the user has explicitly confirmed the parameters
    shown by stage_optimization().  Reads staged parameters from session
    state, dispatches the solver, and writes the solution into the session.

    Returns:
        dict with keys:
          status (str): "success", "feasible", or "error"
          objective_value (float | None)
          num_facilities_selected (int | None)
          solution_summary (str): human-readable result
          error_message (str | None)
          warnings (list[str]): non-fatal warnings (e.g. a road-network fetch
                                failure that triggered the geodesic fallback).
                                The agent should relay these to the user.
          distance_metric_used (str): the metric actually used for this run
                                      (may differ from the requested metric
                                      when the network fallback fires).
    """
    if tool_context is None:
        return {"status": "error", "error_message": "No tool context available."}

    pending = tool_context.state.get("pending_optimization")
    if not pending:
        return {
            "status": "error",
            "error_message": (
                "No staged optimization found. "
                "Call stage_optimization() first, then ask the user to confirm."
            ),
        }

    problem_type = pending["problem_type"]
    parameters = dict(pending.get("parameters", {}))
    constraints = dict(pending.get("constraints", {}))
    distance_metric = pending.get("distance_metric", "network")
    strict_network = bool(pending.get("strict_network", False))

    registry = get_problem_registry()
    if not registry:
        return {"status": "error", "error_message": "Problem registry not available."}

    problem_solver = registry.get_problem(problem_type)
    if not problem_solver:
        return {
            "status": "error",
            "error_message": f"Problem type '{problem_type}' not found in registry.",
        }

    from utils.data_processor import DataProcessor
    processor = DataProcessor()
    data_store = get_data()
    ps = get_problem_state()

    boundary_keys, demand_keys, candidate_keys = _categorise_data(data_store)

    # Build data_dict for solver
    data_dict: dict = {}
    for name in demand_keys:
        data_dict["demand_points"] = data_store[name]
    for name in candidate_keys:
        data_dict["candidate_sites"] = data_store[name]

    # Last-resort: distribute non-boundary data
    if "demand_points" not in data_dict and "candidate_sites" not in data_dict:
        non_boundary = [
            (n, gdf) for n, gdf in data_store.items() if n not in boundary_keys
        ]
        if len(non_boundary) >= 2:
            data_dict["demand_points"] = non_boundary[0][1]
            data_dict["candidate_sites"] = non_boundary[1][1]
        elif len(non_boundary) == 1:
            data_dict["demand_points"] = non_boundary[0][1]

    # Generate candidate sites if missing
    if "demand_points" in data_dict and "candidate_sites" not in data_dict:
        boundary_gdf = next(
            (data_store[n] for n in boundary_keys), None
        )
        sampling_gdf = boundary_gdf if boundary_gdf is not None else data_dict["demand_points"]
        num_sites = get_generated_sites_count()
        seed = get_generated_sites_seed()
        try:
            generated = processor.generate_candidate_sites(
                sampling_gdf, num_sites=num_sites, random_seed=seed
            )
            data_dict["candidate_sites"] = generated
            data_store["generated_candidates"] = generated
            logger.info(
                "confirm_optimization: generated %d candidate sites (seed=%s)",
                num_sites, seed,
            )
        except Exception as exc:
            return {
                "status": "error",
                "error_message": f"Failed to generate candidate sites: {exc}",
            }

    # Auto-detect variant-specific data
    variant = parameters.get("variant", "base")
    if variant == "capacitated" and "capacities" not in parameters:
        if "candidate_sites" in data_dict:
            cap = processor.extract_capacity_data(data_dict["candidate_sites"])
            if cap:
                parameters["capacities"] = cap
            elif "demand_points" in data_dict:
                demand = processor.extract_demand_data(data_dict["demand_points"])
                if demand:
                    n_fac = parameters.get("n_facilities", len(data_dict["candidate_sites"]))
                    avg_cap = sum(demand) / max(n_fac, 1)
                    parameters["capacities"] = [avg_cap] * len(data_dict["candidate_sites"])

    if variant == "budget" and "facility_costs" not in parameters:
        if "candidate_sites" in data_dict:
            costs = processor.extract_cost_data(data_dict["candidate_sites"])
            if costs:
                parameters["facility_costs"] = costs
            else:
                parameters["variant"] = "base" if problem_type == "p-median" else "classical"

    # Add default weights
    if "demand_points" in data_dict:
        data_dict["demand_points"] = processor.add_default_weights(
            data_dict["demand_points"], weight_column="default_weight"
        )

    # Validate required data
    required_data = problem_solver.get_required_data()
    missing = [k for k, v in required_data.items() if v.get("required") and k not in data_dict]
    if missing:
        return {
            "status": "error",
            "error_message": f"Missing required data: {', '.join(missing)}",
        }

    # Fetch road-network graph when distance_metric == "network"
    # On any failure, auto-fall back to geodesic ("euclidean") and surface a
    # warning in the response -- unless strict_network is set, in which case
    # the fetch failure is a hard error.
    network_graph = None
    network_warnings: list = []

    def _network_fallback(reason: str, source: str = "OpenStreetMap") -> Optional[dict]:
        """Emit activity-log event + warning, or return an error dict when strict."""
        try:
            from utils.activity_log import log_event
            log_event(
                "network.fetch",
                "fail",
                detail=reason,
                source=source,
            )
        except Exception:  # activity_log import guarded for test isolation
            pass
        if strict_network:
            return {
                "status": "error",
                "error_message": (
                    f"Road-network distance requested (strict_network=True) but "
                    f"the road graph could not be obtained: {reason}. "
                    "Disable strict_network or switch to 'euclidean' distance."
                ),
            }
        msg = (
            f"Road-network distance unavailable ({reason}); "
            "falling back to geodesic (straight-line) distance for this run."
        )
        network_warnings.append(msg)
        logger.warning("confirm_optimization: %s", msg)
        return None

    if distance_metric == "network":
        nm = get_network_manager()
        if nm is None:
            err = _network_fallback("NetworkManager not in session")
            if err is not None:
                return err
            distance_metric = "euclidean"
        elif not nm.is_osmnx_available():
            err = _network_fallback("osmnx is not installed")
            if err is not None:
                return err
            distance_metric = "euclidean"
        else:
            demand_gdf = data_dict.get("demand_points")
            boundary_gdf = next((data_store[n] for n in boundary_keys), None)
            boundary_polygon = None
            if boundary_gdf is not None and len(boundary_gdf) > 0:
                try:
                    from shapely.ops import unary_union
                    boundary_polygon = unary_union(boundary_gdf.geometry)
                except Exception:
                    boundary_polygon = None
            try:
                G_proj, crs_proj = nm.get_graph(demand_gdf, boundary_polygon)
                network_graph = (G_proj, crs_proj)
            except Exception as exc:
                logger.error(
                    "confirm_optimization: road network fetch error: %s",
                    exc,
                    exc_info=True,
                )
                err = _network_fallback(str(exc))
                if err is not None:
                    return err
                distance_metric = "euclidean"

    if network_graph is not None:
        data_dict["_network_graph"] = network_graph

    # Run solver
    logger.info(
        "confirm_optimization: solving %s with params %s", problem_type, parameters
    )
    start = time.time()
    try:
        solution = problem_solver.solve(
            data=data_dict,
            parameters=parameters,
            constraints=constraints,
            distance_metric=distance_metric,
        )
    except Exception as exc:
        logger.error("confirm_optimization: solver error: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error_message": f"Solver error: {exc}",
            "warnings": list(network_warnings),
            "distance_metric_used": distance_metric,
        }

    elapsed = time.time() - start
    logger.info(
        "confirm_optimization: solved in %.2fs, status=%s", elapsed, solution.get("status")
    )

    # Write solution to Streamlit problem_state
    ps["solution"] = solution
    ps["solution_history"] = ps.get("solution_history", [])
    ps["solution_history"].append(solution)

    # Update ADK session state with summary
    summary_dict = {
        "status": solution.get("status"),
        "objective_value": solution.get("objective_value"),
        "num_facilities": len(solution.get("selected_facilities", [])),
    }
    if tool_context is not None:
        tool_context.state["solution_summary"] = summary_dict
        # Clear pending after successful run
        tool_context.state["pending_optimization"] = None

    # Generate explanation
    explanation = ""
    if solution.get("status") != "error":
        try:
            sig = inspect.signature(problem_solver.explain_solution)
            kwargs = {
                "solution": solution,
                "data": data_dict,
                "detail_level": "standard",
            }
            if "objective_type" in sig.parameters:
                kwargs["objective_type"] = parameters.get("objective", "total")
            explanation = problem_solver.explain_solution(**kwargs)
        except Exception as exc:
            logger.warning("confirm_optimization: explain_solution failed: %s", exc)
            explanation = f"Optimization completed with status: {solution.get('status')}"

    # Forward any network-fetch fallback warnings so the agent can surface them.
    combined_warnings: list = list(network_warnings)
    solver_warnings = solution.get("warnings") if isinstance(solution, dict) else None
    if isinstance(solver_warnings, list):
        combined_warnings.extend(solver_warnings)

    return {
        "status": solution.get("status", "error"),
        "objective_value": solution.get("objective_value"),
        "num_facilities_selected": len(solution.get("selected_facilities", [])),
        "solution_summary": explanation,
        "error_message": solution.get("error"),
        "warnings": combined_warnings,
        "distance_metric_used": distance_metric,
    }
