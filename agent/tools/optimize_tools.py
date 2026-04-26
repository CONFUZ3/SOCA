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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional

from google.adk.tools.tool_context import ToolContext

from config.settings import settings

from .state_bridge import (
    get_data,
    get_problem_state,
    get_problem_registry,
    get_generated_sites_count,
    get_generated_sites_seed,
    get_network_manager,
)

logger = logging.getLogger(__name__)


def _polygon_area_km2(polygon) -> float:
    """Return the polygon area in km² using a WGS84 geodesic computation."""
    try:
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        area_m2, _ = geod.geometry_area_perimeter(polygon)
        return abs(area_m2) / 1_000_000.0
    except Exception:
        return 0.0


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
        from utils.scale_classifier import VALID_POI_CATEGORIES

        if "_facilities_" in key_lower or any(
            key_lower.startswith(cat + "_") for cat in VALID_POI_CATEGORIES
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
    force: bool = False,
    existing_facilities_key: Optional[str] = None,
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
        force: If True, bypass the synthetic-data gate that normally blocks
                        runs when demand or candidates are synthetic
                        fallbacks (HDX failed, no road graph, etc.). The
                        agent must surface the warning to the user and only
                        re-stage with force=True after explicit confirmation.
        existing_facilities_key: Optional dataset key in the data store
                        pointing to a GeoDataFrame of already-placed
                        facilities. When provided, the solver treats those
                        facilities as fixed-open / pre-covering demand.

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
        "force": bool(force),
        "existing_facilities_key": existing_facilities_key,
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


def _apply_existing_facilities(data_dict: dict, parameters: dict) -> dict:
    """Pre-process the solver inputs to account for already-placed facilities.

    Coverage models (MCLP / LSCP) — i.e. anything where ``service_radius`` is
    set — drop demand points already within radius of any existing facility.
    Their count and weight are returned in ``info`` for post-solve reporting.

    For non-coverage models (P-Median / P-Center) we record existing-facility
    statistics for the result metrics but do not drop demand: the new
    facilities are optimised as if existing weren't present, which over-counts
    the new objective by the contribution of already-served demand. This is a
    documented MVP simplification — full fixed-open MIP integration would
    require touching each solver's IP code.

    Mutates *data_dict* in place when residualising demand. Returns ``info``
    (possibly empty) suitable for merging into ``result["metrics"]``.
    """
    info: dict = {}
    existing = data_dict.get("existing_facilities")
    demand_gdf = data_dict.get("demand_points")
    if existing is None or len(existing) == 0:
        return info
    if demand_gdf is None or len(demand_gdf) == 0:
        return info

    try:
        from utils.distance_calculator import DistanceCalculator
        import numpy as _np
        import pandas as _pd
    except Exception as exc:
        logger.warning("existing-facilities preprocess: import failed (%s)", exc)
        return info

    try:
        dist = DistanceCalculator().calculate_distance_matrix(
            demand_gdf, existing, metric="euclidean"
        )
        nearest = dist.min(axis=1)
    except Exception as exc:
        logger.warning("existing-facilities preprocess: distance calc failed (%s)", exc)
        return info

    info["existing_count"] = int(len(existing))
    info["existing_min_distance_avg"] = float(nearest.mean())

    coverage_radius = parameters.get("service_radius")
    if coverage_radius is not None and float(coverage_radius) > 0:
        covered_mask = nearest <= float(coverage_radius)
        n_pre = int(covered_mask.sum())
        if n_pre > 0:
            weights = None
            for col in ("default_weight", "weight", "population", "demand"):
                if col in demand_gdf.columns:
                    try:
                        weights = _pd.to_numeric(demand_gdf[col], errors="coerce").fillna(1.0).values
                        break
                    except Exception:
                        pass
            if weights is None:
                weights = _np.ones(len(demand_gdf))
            info["pre_covered_count"] = n_pre
            info["pre_covered_demand_weight"] = float(weights[covered_mask].sum())
            residual = demand_gdf[~covered_mask].reset_index(drop=True)
            data_dict["demand_points"] = residual
            logger.info(
                "existing-facilities: dropped %d/%d demand points already covered "
                "(weight=%.1f) by %d existing facilities",
                n_pre, len(demand_gdf), info["pre_covered_demand_weight"],
                info["existing_count"],
            )
    return info


def _gdf_data_source(gdf) -> Optional[str]:
    """Best-effort read of a GeoDataFrame's data_source stamp."""
    try:
        if gdf is None or len(gdf) == 0 or "data_source" not in gdf.columns:
            return None
        return str(gdf["data_source"].iloc[0])
    except Exception:
        return None


def _check_synthetic_data_gate(data_dict: dict, force: bool) -> Optional[dict]:
    """Block runs when demand or candidates are synthetic, unless force=True.

    Reads the ``data_source`` column stamped by fetchers and the candidate
    generator. Returns a structured warning dict if the run should be
    blocked, or None to proceed.

    Failure modes: missing ``data_source`` column is treated as
    non-synthetic so that user-uploaded GDFs aren't gated unnecessarily.
    """
    if force:
        return None

    synthetic_layers = []
    reasons = []

    demand_src = _gdf_data_source(data_dict.get("demand_points"))
    if demand_src and "synthetic" in demand_src.lower():
        synthetic_layers.append("demand")
        reasons.append(
            f"Demand points are synthetic ({demand_src}). "
            "Population data could not be fetched (HDX timeout or country "
            "unresolved); a uniform grid was substituted."
        )

    cand_src = _gdf_data_source(data_dict.get("candidate_sites"))
    if cand_src and "synthetic" in cand_src.lower():
        synthetic_layers.append("candidates")
        reasons.append(
            f"Candidate sites are synthetic ({cand_src}). "
            "No road network was available for the AOI; random points were "
            "sampled inside the demand convex hull."
        )

    if not synthetic_layers:
        return None

    msg = (
        "Optimization blocked: "
        + " / ".join(synthetic_layers)
        + " layer(s) are synthetic fallbacks. "
        + " ".join(reasons)
        + " To proceed anyway, re-run with force=True."
    )
    return {
        "status": "warning",
        "blocked_on": "synthetic_data",
        "synthetic_layers": synthetic_layers,
        "reason": " ".join(reasons),
        "user_actions": ["retry_with_better_boundary", "force_with_synthetic"],
        "warnings": [msg],
        "solution_summary": msg,
        "objective_value": None,
        "num_facilities_selected": 0,
        "error_message": None,
    }


def _prepare_solver_inputs(
    data_store: dict,
    boundary_keys: set,
    demand_keys: set,
    candidate_keys: set,
    problem_type: str,
    problem_solver,
    parameters: dict,
    existing_facilities_key: Optional[str] = None,
) -> tuple:
    """Build data_dict and finalise parameters for the solver.

    Returns (data_dict, updated_parameters, error_dict_or_None).
    """
    from utils.data_processor import DataProcessor
    processor = DataProcessor()

    data_dict: dict = {}
    for name in demand_keys:
        data_dict["demand_points"] = data_store[name]
    for name in candidate_keys:
        data_dict["candidate_sites"] = data_store[name]

    if "demand_points" not in data_dict and "candidate_sites" not in data_dict:
        non_boundary = [
            (n, gdf) for n, gdf in data_store.items() if n not in boundary_keys
        ]
        if len(non_boundary) >= 2:
            data_dict["demand_points"] = non_boundary[0][1]
            data_dict["candidate_sites"] = non_boundary[1][1]
        elif len(non_boundary) == 1:
            data_dict["demand_points"] = non_boundary[0][1]

    if "demand_points" in data_dict and "candidate_sites" not in data_dict:
        boundary_gdf = next((data_store[n] for n in boundary_keys), None)
        sampling_gdf = (
            boundary_gdf if boundary_gdf is not None else data_dict["demand_points"]
        )
        num_sites = get_generated_sites_count()
        seed = get_generated_sites_seed()
        # Derive boundary polygon for road-network candidate sampling, and
        # pass the cached NetworkManager so the same graph is reused.
        boundary_polygon = None
        if boundary_gdf is not None and len(boundary_gdf) > 0:
            try:
                from shapely.ops import unary_union
                boundary_polygon = unary_union(boundary_gdf.geometry)
            except Exception:
                boundary_polygon = None
        try:
            generated = processor.generate_candidate_sites(
                sampling_gdf,
                num_sites=num_sites,
                random_seed=seed,
                boundary_polygon=boundary_polygon,
                network_manager=get_network_manager(),
            )
            data_dict["candidate_sites"] = generated
            data_store["generated_candidates"] = generated
            logger.info(
                "confirm_optimization: generated %d candidate sites "
                "(seed=%s, data_source=%s)",
                len(generated), seed, _gdf_data_source(generated),
            )
        except Exception as exc:
            return None, parameters, {
                "status": "error",
                "error_message": f"Failed to generate candidate sites: {exc}",
            }

    # Plumb existing facilities (Priority 4) — looked up by key in the data
    # store. Validated as a non-empty GDF; otherwise silently ignored so the
    # solver runs without it.
    if existing_facilities_key and existing_facilities_key in data_store:
        existing_gdf = data_store[existing_facilities_key]
        try:
            if existing_gdf is not None and len(existing_gdf) > 0:
                data_dict["existing_facilities"] = existing_gdf
                logger.info(
                    "confirm_optimization: existing_facilities loaded from "
                    "key '%s' (%d rows)", existing_facilities_key, len(existing_gdf)
                )
        except Exception as exc:
            logger.warning(
                "confirm_optimization: existing_facilities key '%s' invalid "
                "(%s); ignoring.", existing_facilities_key, exc
            )

    variant = parameters.get("variant", "base")
    if variant == "capacitated" and "capacities" not in parameters:
        if "candidate_sites" in data_dict:
            cap = processor.extract_capacity_data(data_dict["candidate_sites"])
            if cap:
                parameters["capacities"] = cap
            elif "demand_points" in data_dict:
                demand = processor.extract_demand_data(data_dict["demand_points"])
                if demand:
                    n_fac = parameters.get(
                        "n_facilities", len(data_dict["candidate_sites"])
                    )
                    avg_cap = sum(demand) / max(n_fac, 1)
                    parameters["capacities"] = [avg_cap] * len(
                        data_dict["candidate_sites"]
                    )

    if variant == "budget" and "facility_costs" not in parameters:
        if "candidate_sites" in data_dict:
            costs = processor.extract_cost_data(data_dict["candidate_sites"])
            if costs:
                parameters["facility_costs"] = costs
            else:
                parameters["variant"] = (
                    "base" if problem_type == "p-median" else "classical"
                )

    if "demand_points" in data_dict:
        data_dict["demand_points"] = processor.add_default_weights(
            data_dict["demand_points"], weight_column="default_weight"
        )

    required_data = problem_solver.get_required_data()
    missing = [
        k for k, v in required_data.items()
        if v.get("required") and k not in data_dict
    ]
    if missing:
        return None, parameters, {
            "status": "error",
            "error_message": f"Missing required data: {', '.join(missing)}",
        }

    return data_dict, parameters, None


def _fetch_network_graph(
    distance_metric: str,
    strict_network: bool,
    data_dict: dict,
    boundary_keys: set,
    data_store: dict,
) -> tuple:
    """Fetch OSM road-network graph for network-distance solves.

    Returns (network_graph_or_None, resolved_metric, warnings, error_or_None).
    error_or_None is non-None only when strict_network=True and fetch fails.
    """
    if distance_metric != "network":
        return None, distance_metric, [], None

    network_warnings: list = []
    nm = get_network_manager()

    def _fallback(reason: str, source: str = "OpenStreetMap"):
        try:
            from utils.activity_log import log_event
            log_event("network.fetch", "fail", detail=reason, source=source)
        except Exception:
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

    if nm is None:
        err = _fallback("NetworkManager not in session")
        return None, "euclidean", network_warnings, err
    if not nm.is_osmnx_available():
        err = _fallback("osmnx is not installed")
        return None, "euclidean", network_warnings, err

    demand_gdf = data_dict.get("demand_points")
    boundary_gdf = next((data_store[n] for n in boundary_keys), None)
    boundary_polygon = None
    if boundary_gdf is not None and len(boundary_gdf) > 0:
        try:
            from shapely.ops import unary_union
            boundary_polygon = unary_union(boundary_gdf.geometry)
        except Exception:
            boundary_polygon = None

    max_area_km2 = float(getattr(settings, "NETWORK_FETCH_MAX_AREA_KM2", 10_000.0))
    if not strict_network and boundary_polygon is not None and max_area_km2 > 0:
        aoi_area = _polygon_area_km2(boundary_polygon)
        if aoi_area > max_area_km2:
            reason = (
                f"AOI area {aoi_area:,.0f} km² exceeds "
                f"NETWORK_FETCH_MAX_AREA_KM2={max_area_km2:,.0f} km²"
            )
            _fallback(reason)
            return None, "euclidean", network_warnings, None

    fetch_timeout = float(getattr(settings, "NETWORK_FETCH_TIMEOUT", 45.0))
    try:
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="netfetch"
        ) as pool:
            future = pool.submit(nm.get_graph, demand_gdf, boundary_polygon)
            try:
                G_proj, crs_proj = future.result(timeout=fetch_timeout)
                return (G_proj, crs_proj), distance_metric, network_warnings, None
            except FutureTimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"road-network fetch exceeded {fetch_timeout:.0f}s wall-clock"
                )
    except Exception as exc:
        logger.error(
            "confirm_optimization: road network fetch error: %s", exc, exc_info=True
        )
        err = _fallback(str(exc))
        return None, "euclidean", network_warnings, err


def _run_solver_with_timeout(
    problem_solver,
    data_dict: dict,
    parameters: dict,
    constraints: dict,
    distance_metric: str,
    wall_clock: float,
    problem_type: str,
    network_warnings: list,
) -> tuple:
    """Dispatch solver in a thread pool with a hard wall-clock limit.

    Returns (solution, elapsed_s, error_result_or_None).
    """
    try:
        from utils.activity_log import timed as _solver_timed, log_event
        solver_ctx = _solver_timed(
            "solver.run",
            source=problem_type,
            detail=f"variant={parameters.get('variant', 'base')}, metric={distance_metric}",
        )
    except Exception:
        solver_ctx = None
        log_event = None  # type: ignore[assignment]

    logger.info(
        "confirm_optimization: solving %s with params %s", problem_type, parameters
    )
    start = time.time()

    def _run():
        return problem_solver.solve(
            data=data_dict,
            parameters=parameters,
            constraints=constraints,
            distance_metric=distance_metric,
        )

    def _submit():
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="solver"
        ) as pool:
            future = pool.submit(_run)
            try:
                return future.result(timeout=wall_clock)
            except FutureTimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"solver exceeded wall-clock budget of {wall_clock:.0f}s"
                )

    try:
        if solver_ctx is not None:
            with solver_ctx:
                solution = _submit()
        else:
            solution = _submit()
    except TimeoutError as exc:
        logger.error("confirm_optimization: %s", exc)
        if log_event:
            try:
                log_event("solver.run", "fail", str(exc), source=problem_type)
            except Exception:
                pass
        warn = (
            f"Solver stopped after {wall_clock:.0f}s wall-clock limit. "
            "Try a smaller AOI, switch to 'euclidean' distance, or reduce "
            "demand density."
        )
        return None, time.time() - start, {
            "status": "timeout",
            "objective_value": None,
            "num_facilities_selected": 0,
            "solution_summary": warn,
            "error_message": str(exc),
            "warnings": list(network_warnings) + [warn],
            "distance_metric_used": distance_metric,
        }
    except Exception as exc:
        logger.error("confirm_optimization: solver error: %s", exc, exc_info=True)
        return None, time.time() - start, {
            "status": "error",
            "error_message": f"Solver error: {exc}",
            "warnings": list(network_warnings),
            "distance_metric_used": distance_metric,
        }

    elapsed = time.time() - start
    logger.info(
        "confirm_optimization: solved in %.2fs, status=%s",
        elapsed, solution.get("status"),
    )
    return solution, elapsed, None


def _enrich_and_explain(
    solution: dict,
    problem_solver,
    parameters: dict,
    data_dict: dict,
    problem_type: str,
    distance_metric: str,
    network_warnings: list,
    elapsed: float,
    existing_info: Optional[dict] = None,
    boundary_polygon=None,
) -> tuple:
    """Augment raw solver output with run context and generate explanation.

    Adds equity metrics (Priority 6), merges existing-facilities pre-solve
    info into ``metrics`` (Priority 4), writes a reproducibility log entry
    (Priority 7), and returns ``(enriched_solution, explanation_text)``.
    """
    # Merge existing-facilities pre-solve info into solver metrics.
    if existing_info:
        try:
            solution.setdefault("metrics", {})
            solution["metrics"].update(existing_info)
        except Exception as exc:
            logger.warning("enrich: existing_info merge failed (%s)", exc)

    # Equity metrics (Priority 6) — computed regardless of solver.
    try:
        from utils.equity_metrics import compute_equity_metrics
        from utils.distance_calculator import DistanceCalculator
        import pandas as _pd
        demand_gdf = data_dict.get("demand_points")
        cand_gdf = data_dict.get("candidate_sites")
        if demand_gdf is not None and cand_gdf is not None and len(demand_gdf) > 0:
            ngraph = data_dict.get("_network_graph")
            try:
                D = DistanceCalculator().calculate_distance_matrix(
                    demand_gdf, cand_gdf,
                    metric=distance_metric,
                    network_graph=ngraph,
                )
            except Exception:
                D = DistanceCalculator().calculate_distance_matrix(
                    demand_gdf, cand_gdf, metric="euclidean"
                )
            weights = None
            for col in ("default_weight", "weight", "population", "demand"):
                if col in demand_gdf.columns:
                    try:
                        weights = _pd.to_numeric(demand_gdf[col], errors="coerce").fillna(1.0).values
                        break
                    except Exception:
                        pass
            equity = compute_equity_metrics(
                D,
                solution.get("selected_facilities", []) or [],
                solution.get("assignments") or {},
                demand_weights=weights,
                coverage_radius=parameters.get("service_radius"),
            )
            solution["equity_metrics"] = equity
    except Exception as exc:
        logger.warning("enrich: equity computation failed (%s)", exc)

    # Reproducibility log (Priority 7).
    try:
        from utils.repro_logger import ReproducibilityLogger, build_run_payload, get_seed
        payload = build_run_payload(
            boundary_polygon=boundary_polygon,
            demand_gdf=data_dict.get("demand_points"),
            candidates_gdf=data_dict.get("candidate_sites"),
            distance_method=distance_metric,
            solver=(solution.get("solver_details") or {}).get("solver") or solution.get("solver"),
            solver_params=parameters,
            objective_value=solution.get("objective_value"),
            selected_facility_ids=solution.get("selected_facilities") or [],
            random_seed=get_seed(),
            extra={
                "problem_type": problem_type,
                "variant": parameters.get("variant", "base"),
                "elapsed_s": elapsed,
                "existing_facilities": existing_info or {},
                "equity_metrics": solution.get("equity_metrics"),
            },
        )
        log_path = ReproducibilityLogger().log_run(payload)
        solution["repro_log_path"] = str(log_path)
    except Exception as exc:
        logger.warning("enrich: repro log failed (%s)", exc)

    try:
        existing = solution.get("warnings") if isinstance(solution, dict) else None
        merged = list(network_warnings)
        if isinstance(existing, list):
            merged.extend(existing)
        solver_name = (
            (solution.get("solver_details") or {}).get("solver")
            if isinstance(solution, dict)
            else None
        )
        solution["distance_metric_used"] = distance_metric
        solution["warnings"] = merged
        solution["problem_type"] = problem_type
        solution["variant"] = parameters.get("variant", "base")
        solution["solver_time_seconds"] = float(
            solution.get("solution_time", elapsed) or elapsed
        )
        solution["solver"] = solver_name or solution.get("solver") or "unknown"
        sr = parameters.get("service_radius")
        if sr is not None:
            try:
                solution["service_radius_m"] = float(sr)
            except (TypeError, ValueError):
                pass
    except Exception as exc:
        logger.warning("confirm_optimization: solution enrichment failed: %s", exc)

    explanation = ""
    if solution.get("status") != "error":
        try:
            sig = inspect.signature(problem_solver.explain_solution)
            kwargs: dict = {
                "solution": solution,
                "data": data_dict,
                "detail_level": "standard",
            }
            if "objective_type" in sig.parameters:
                kwargs["objective_type"] = parameters.get("objective", "total")
            explanation = problem_solver.explain_solution(**kwargs)
        except Exception as exc:
            logger.warning(
                "confirm_optimization: explain_solution failed: %s", exc
            )
            explanation = (
                f"Optimization completed with status: {solution.get('status')}"
            )

    return solution, explanation


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
          warnings (list[str]): non-fatal warnings (e.g. road-network fallback).
          distance_metric_used (str): the metric actually used for this run.
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
    force = bool(pending.get("force", False))
    existing_facilities_key = pending.get("existing_facilities_key")

    registry = get_problem_registry()
    if not registry:
        return {"status": "error", "error_message": "Problem registry not available."}
    problem_solver = registry.get_problem(problem_type)
    if not problem_solver:
        return {
            "status": "error",
            "error_message": f"Problem type '{problem_type}' not found in registry.",
        }

    data_store = get_data()
    ps = get_problem_state()
    boundary_keys, demand_keys, candidate_keys = _categorise_data(data_store)

    data_dict, parameters, err = _prepare_solver_inputs(
        data_store, boundary_keys, demand_keys, candidate_keys,
        problem_type, problem_solver, parameters,
        existing_facilities_key=existing_facilities_key,
    )
    if err is not None:
        return err

    # Existing facilities pre-process (Priority 4). Drops demand already
    # covered by existing facilities for coverage models; records stats
    # for non-coverage models. Mutates data_dict in place.
    existing_info = _apply_existing_facilities(data_dict, parameters)

    # Distance-metric auto-downgrade for very large AOIs (Priority 3
    # heuristic). If the user/agent did not explicitly pick "euclidean"
    # and the boundary exceeds NETWORK_AUTO_EUCLIDEAN_AREA_KM2, drop to
    # geodesic to avoid multi-minute Overpass downloads on regional AOIs.
    auto_threshold = float(getattr(settings, "NETWORK_AUTO_EUCLIDEAN_AREA_KM2", 2_000.0))
    if (
        distance_metric == "network"
        and not strict_network
        and auto_threshold > 0
    ):
        boundary_gdf_for_area = next((data_store[n] for n in boundary_keys), None)
        if boundary_gdf_for_area is not None and len(boundary_gdf_for_area) > 0:
            try:
                from shapely.ops import unary_union
                _bp = unary_union(boundary_gdf_for_area.geometry)
                _aoi_km2 = _polygon_area_km2(_bp)
                if _aoi_km2 > auto_threshold:
                    msg = (
                        f"AOI area {_aoi_km2:,.0f} km² exceeds auto-downgrade "
                        f"threshold {auto_threshold:,.0f} km²; using geodesic "
                        "(euclidean) distance instead of road-network."
                    )
                    logger.warning("confirm_optimization: %s", msg)
                    distance_metric = "euclidean"
                    # Stash for warning aggregation below.
                    parameters.setdefault("_auto_metric_warnings", []).append(msg)
            except Exception:
                pass

    network_graph, distance_metric, network_warnings, err = _fetch_network_graph(
        distance_metric, strict_network, data_dict, boundary_keys, data_store,
    )
    if err is not None:
        return err
    # Surface auto-downgrade warning into the network_warnings list.
    for _w in parameters.pop("_auto_metric_warnings", []):
        if _w not in network_warnings:
            network_warnings.insert(0, _w)

    # Synthetic-data gate (Priority 2). Block silently-fallback runs unless
    # the agent has explicitly confirmed with force=True. Runs AFTER the
    # network fetch so that strict_network errors take precedence.
    gate = _check_synthetic_data_gate(data_dict, force)
    if gate is not None:
        logger.warning(
            "confirm_optimization: synthetic-data gate triggered (%s)",
            gate.get("synthetic_layers"),
        )
        return gate
    if network_graph is not None:
        data_dict["_network_graph"] = network_graph

    # Model-size guard: shorten MIP budget for large instances so the GA
    # fallback kicks in before Python model-building dominates.
    try:
        n_demand = len(data_dict.get("demand_points", []))
        n_cand = len(data_dict.get("candidate_sites", []))
    except Exception:
        n_demand = n_cand = 0
    model_size = n_demand * n_cand
    size_limit = int(getattr(settings, "MIP_MODEL_SIZE_LIMIT", 300_000))
    if size_limit > 0 and model_size > size_limit:
        msg = (
            f"Large model detected ({n_demand:,} demand × {n_cand:,} candidates "
            f"= {model_size:,} pairs, limit {size_limit:,}). "
            "Shortening MIP budget and relying on the genetic-algorithm fallback."
        )
        logger.warning("confirm_optimization: %s", msg)
        network_warnings.append(msg)
        parameters.setdefault("fallback_time_limit_seconds", 10.0)

    parameters.setdefault(
        "fallback_time_limit_seconds", float(settings.SOLVER_MIP_TIME_LIMIT)
    )
    parameters.setdefault(
        "ga_time_budget_seconds", float(settings.SOLVER_GA_TIME_LIMIT)
    )

    wall_clock = float(settings.SOLVER_WALL_CLOCK_TIMEOUT)
    solution, elapsed, err = _run_solver_with_timeout(
        problem_solver, data_dict, parameters, constraints,
        distance_metric, wall_clock, problem_type, network_warnings,
    )
    if err is not None:
        return err

    # Resolve boundary polygon for the repro log payload.
    _boundary_polygon_for_log = None
    _bgdf = next((data_store[n] for n in boundary_keys), None)
    if _bgdf is not None and len(_bgdf) > 0:
        try:
            from shapely.ops import unary_union
            _boundary_polygon_for_log = unary_union(_bgdf.geometry)
        except Exception:
            _boundary_polygon_for_log = None

    solution, explanation = _enrich_and_explain(
        solution, problem_solver, parameters, data_dict,
        problem_type, distance_metric, network_warnings, elapsed,
        existing_info=existing_info,
        boundary_polygon=_boundary_polygon_for_log,
    )

    ps["solution"] = solution
    ps["solution_history"] = ps.get("solution_history", [])
    ps["solution_history"].append(solution)

    if tool_context is not None:
        tool_context.state["solution_summary"] = {
            "status": solution.get("status"),
            "objective_value": solution.get("objective_value"),
            "num_facilities": len(solution.get("selected_facilities", [])),
        }
        tool_context.state["pending_optimization"] = None

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
        "equity_metrics": solution.get("equity_metrics"),
        "repro_log_path": solution.get("repro_log_path"),
        "existing_facilities_info": existing_info or None,
    }
