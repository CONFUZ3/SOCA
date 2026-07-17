from .base_solver import SpatialOptimizationProblem
from typing import Dict, List, Any, Optional
import geopandas as gpd
import numpy as np
import math
import time
import logging

from utils.heuristics.genetic_solver import GAConfig, LSCPGeneticSolver

logger = logging.getLogger(__name__)

class LSCPSolver(SpatialOptimizationProblem):
    """
    Location Set Covering Problem (LSCP) Solver
    
    Minimizes the number of facilities needed to cover all demand within a distance threshold.
    """
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Location Set Covering Problem (LSCP)",
            "short_name": "lscp",
            "category": "coverage minimization",
            "description": "Locate the minimum number of facilities needed to ensure all demand points are covered within a specified service distance. Optimal when full coverage is required and you want to minimize costs/facilities.",
            "mathematical_formulation": """
Minimize: Σⱼ xⱼ

Subject to:
- Σⱼ∈Nᵢ xⱼ ≥ 1, ∀i  (each demand must be covered)
- xⱼ ∈ {0,1}

Where:
- xⱼ = 1 if facility located at j
- Nᵢ = set of candidates within threshold distance of demand i
            """,
            "academic_refs": [
                "Toregas, C., Swain, R., ReVelle, C., & Bergman, L. (1971). The location of emergency service facilities. Operations Research, 19(6), 1363-1373.",
                "Church, R. L., & ReVelle, C. (1976). Theoretical and computational links between the p-median, location set-covering, and the maximal covering location problem. Geographical Analysis, 8(4), 406-415.",
                "Daskin, M. S. (2013). Network and discrete location: models, algorithms, and applications. John Wiley & Sons."
            ],
            "complexity": "NP-hard (Set Cover reduction)",
            "typical_use_cases": [
                "Emergency service minimum deployment",
                "Minimum infrastructure for full coverage",
                "Cell tower placement with coverage requirements",
                "Public service facility budgeting",
                "Sensor network minimum deployment"
            ],
            "keywords": [
                "lscp", "set cover", "minimize facilities", "full coverage",
                "cover all", "minimum number", "complete coverage"
            ],
            "variants": ["base", "backup", "conditional", "probabilistic", "partial"]
        }
    
    def get_conversation_prompts(self) -> Dict[str, Any]:
        return {
            "problem_detection": [
                "minimize facilities", "minimum number", "full coverage",
                "cover all", "lscp", "set cover", "complete coverage",
                "minimum cost", "fewest facilities"
            ],
            "parameter_questions": [
                {
                    "param": "service_radius",
                    "question": "What is the maximum service distance/radius for coverage?",
                    "type": "float",
                    "validation": "Must be a positive number",
                    "help": "Demand is covered if within this distance from a facility"
                }
            ],
            "constraint_suggestions": [
                "Would you like to specify facilities that must be included?",
                "Should any facilities be excluded from consideration?",
                "Do you have a maximum budget (upper limit on number of facilities)?"
            ],
            "explanation_template": "The LSCP solution requires {n_facilities} facilities to achieve full coverage within radius {service_radius}."
        }
    
    def get_required_data(self) -> Dict[str, Dict[str, Any]]:
        return {
            "demand_points": {
                "required": True,
                "description": "Points representing demand locations",
                "required_fields": [],
                "optional_fields": ["demand", "weight", "population"],
                "geometry_type": "Point"
            },
            "candidate_sites": {
                "required": True,
                "description": "Potential facility locations",
                "required_fields": [],
                "optional_fields": [],
                "geometry_type": "Point"
            }
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        if "service_radius" not in params:
            return False, "Missing required parameter: service_radius"
        
        if not isinstance(params["service_radius"], (int, float)) or params["service_radius"] <= 0:
            return False, "service_radius must be a positive number"

        ok, err = self._validate_facility_sets(params)
        if not ok:
            return False, err

        variant = params.get("variant", "base")
        allowed = {"base", "backup", "conditional", "probabilistic", "partial"}
        if variant not in allowed:
            return False, f"Unknown variant '{variant}'. Allowed: {sorted(allowed)}"

        if variant == "backup":
            k = params.get("k_coverage", 2)
            if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                return False, "k_coverage must be an integer >= 1 for the backup variant"

        if variant == "probabilistic":
            alpha = params.get("coverage_reliability", 0.95)
            if not isinstance(alpha, (int, float)) or not (0.0 < alpha < 1.0):
                return False, "coverage_reliability must satisfy 0 < alpha < 1 for the probabilistic variant"

        if variant == "partial":
            fraction = params.get("coverage_fraction", 0.95)
            if not isinstance(fraction, (int, float)) or not (0.0 < fraction <= 1.0):
                return False, "coverage_fraction must satisfy 0 < f <= 1 for the partial variant"

        if variant == "conditional":
            existing = params.get("existing_facilities")
            if not existing:
                return False, "existing_facilities must be a non-empty list for the conditional variant"

        return True, None

    def _normalize_reliability(self, reliability: Optional[Any], n_candidates: int) -> np.ndarray:
        """Return a length-n reliability array in [0,1].

        Accepts None, scalar, list, or numpy array. If None, returns ones.
        If scalar provided, broadcasts to length n. If array-like provided,
        validates length and value range.
        """
        if reliability is None:
            return np.ones(n_candidates, dtype=float)
        # Convert to numpy array if not already
        if np.isscalar(reliability):
            val = float(reliability)
            if not (0.0 <= val <= 1.0):
                raise ValueError("facility_reliability scalar must be between 0 and 1")
            return np.full(n_candidates, val, dtype=float)
        arr = np.asarray(reliability, dtype=float)
        if arr.ndim == 0:
            # 0-d array: treat as scalar
            val = float(arr)
            if not (0.0 <= val <= 1.0):
                raise ValueError("facility_reliability scalar must be between 0 and 1")
            return np.full(n_candidates, val, dtype=float)
        if arr.shape[0] != n_candidates:
            raise ValueError("Length of facility_reliability must match number of candidate sites")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError("facility_reliability values must be between 0 and 1")
        return arr

    def solve(
        self,
        data: Dict[str, gpd.GeoDataFrame],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        distance_metric: str = "network"
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            from utils.distance_calculator import DistanceCalculator
            
            demand_gdf = data.get('demand_points')
            candidate_gdf = data.get('candidate_sites')
            
            if demand_gdf is None or candidate_gdf is None:
                raise ValueError("Both demand_points and candidate_sites are required")
            
            service_radius = parameters['service_radius']
            # Get explicit unit from parameters (default to 'm' if not specified)
            service_radius_unit = parameters.get('service_radius_unit', 'm')

            variant = parameters.get('variant', 'base')
            
            # Calculate coverage matrix
            dist_calc = DistanceCalculator()

            # Get unit conversion info
            unit_info = dist_calc.get_unit_info(service_radius, service_radius_unit)

            network_graph = data.get('_network_graph')
            coverage_matrix = dist_calc.calculate_coverage_matrix(
                demand_gdf, candidate_gdf,
                threshold=service_radius,
                metric=distance_metric,
                unit=service_radius_unit,
                network_graph=network_graph,
            )

            distance_matrix = dist_calc.calculate_distance_matrix(
                demand_gdf, candidate_gdf, metric=distance_metric, network_graph=network_graph
            )
            
            n_candidates = coverage_matrix.shape[1]

            # Variant-specific extras
            k = int(parameters.get('k_coverage', 2))
            weights = self._extract_weights(demand_gdf, parameters)
            reliability = self._normalize_reliability(parameters.get('facility_reliability'), n_candidates)
            alpha = float(parameters.get('coverage_reliability', 0.95))
            fraction = float(parameters.get('coverage_fraction', 0.95))
            existing = sorted(set(int(v) for v in (parameters.get('existing_facilities') or [])))

            # Check feasibility
            # For 'backup', a demand is uncoverable if fewer than k candidates cover it.
            # For 'partial', uncoverable demand is allowed (those demands simply stay uncovered).
            uncoverable = []
            for i in range(len(demand_gdf)):
                if variant == "backup":
                    if int(coverage_matrix[i, :].sum()) < k:
                        uncoverable.append(i)
                else:
                    if not np.any(coverage_matrix[i, :]):
                        uncoverable.append(i)

            constraints = self._merge_facility_set_constraints(constraints, parameters)

            if uncoverable and variant != "partial":
                if variant == "backup":
                    error_msg = (
                        f"Infeasible: {len(uncoverable)} demand points cannot be covered by at least "
                        f"k={k} facilities within service radius {service_radius}. Consider increasing the "
                        f"service radius, lowering k_coverage, or adding more candidate sites."
                    )
                else:
                    error_msg = (
                        f"Infeasible: {len(uncoverable)} demand points cannot be covered with service radius "
                        f"{service_radius}. Consider increasing the service radius or adding more candidate sites."
                    )
                return {
                    "status": "infeasible",
                    "error": error_msg,
                    "uncoverable_points": uncoverable,
                    "solution_time": time.time() - start_time
                }

            fallback_time_limit = float(parameters.get('fallback_time_limit_seconds', 60.0))
            ga_time_budget = float(parameters.get('ga_time_budget_seconds', 60.0))
            logger.info(f"LSCP: Fallback time limit set to {fallback_time_limit:.2f} seconds, GA time budget: {ga_time_budget:.2f} seconds")
            
            mip_start = time.time()
            solution = self._solve_mip(
                coverage_matrix,
                constraints,
                time_limit_seconds=fallback_time_limit,
                variant=variant,
                k=k,
                weights=weights,
                reliability=reliability,
                alpha=alpha,
                fraction=fraction,
                existing=existing
            )
            mip_elapsed = time.time() - mip_start

            timed_out_flag = bool(solution.get('solver_details', {}).get('timed_out', False))
            ga_needed = timed_out_flag or (
                fallback_time_limit > 0 and mip_elapsed >= max(0.1, 0.95 * fallback_time_limit)
            )
            logger.info(f"LSCP timeout check: mip_elapsed={mip_elapsed:.2f}s, fallback_limit={fallback_time_limit:.2f}s, timed_out_flag={timed_out_flag}, ga_needed={ga_needed}")
            if ga_needed:
                ga_cfg = GAConfig(time_limit_seconds=ga_time_budget)
                ga_solver = LSCPGeneticSolver(ga_cfg)
                if ga_solver.supports_variant(variant):
                    logger.info("LSCP: Falling back to Genetic Algorithm")
                    logger.info(f"LSCP: MIP solver status: {solution.get('status', 'unknown')}, objective: {solution.get('objective_value', 'N/A')}")
                    incumbent_mask = None
                    if solution.get('selected_facilities'):
                        incumbent_mask = np.zeros(coverage_matrix.shape[1], dtype=np.int8)
                        for idx in solution['selected_facilities']:
                            if 0 <= idx < coverage_matrix.shape[1]:
                                incumbent_mask[idx] = 1
                    logger.info(f"LSCP: Starting GA with time budget: {ga_time_budget:.2f} seconds")
                    ga_result = ga_solver.solve(
                        coverage_matrix=coverage_matrix,
                        distance_matrix=distance_matrix,
                        time_budget_seconds=ga_time_budget,
                        initial_solution=incumbent_mask,
                        variant=variant,
                        k_coverage=k,
                        demand_weights=weights,
                        reliability=reliability,
                        coverage_reliability=alpha,
                        coverage_fraction=fraction,
                        existing=existing,
                    )
                    logger.info(f"LSCP: GA completed with status: {ga_result.get('status', 'unknown')}, objective: {ga_result.get('objective_value', 'N/A')}")
                    ga_details = {
                        **ga_result.get("solver_details", {}),
                        "fallback_from": solution.get('solver_details', {}).get('solver', 'mip'),
                        "fallback_reason": "time_limit"
                    }
                    ga_warnings = list(solution.get('warnings', []))
                    if ga_result.get("status") == "approximate":
                        ga_warnings.append(
                            f"GA fallback returned an approximate solution that may marginally "
                            f"violate the '{variant}' constraint."
                        )
                    solution = {
                        "status": ga_result.get("status", "feasible"),
                        "objective_value": len(ga_result["selected_facilities"]),
                        "selected_facilities": ga_result["selected_facilities"],
                        "assignments": ga_result["assignments"],
                        "solver_details": ga_details,
                        "warnings": ga_warnings,
                    }
                else:
                    logger.info(f"LSCP: GA fallback not available for variant '{variant}'; keeping MIP incumbent")
                    if timed_out_flag:
                        solution.setdefault('warnings', []).append(
                            f"MIP time limit reached for the '{variant}' LSCP variant; returning best incumbent "
                            f"(no GA fallback for this variant)."
                        )
            else:
                logger.info(f"LSCP: MIP solver completed successfully within time limit, no fallback needed. Status: {solution.get('status', 'unknown')}, objective: {solution.get('objective_value', 'N/A')}")

            # Calculate metrics
            metrics = self._calculate_metrics(
                coverage_matrix, distance_matrix,
                solution['selected_facilities'],
                service_radius,
                service_radius_unit=service_radius_unit
            )
            
            solution_time = time.time() - start_time
            
            return {
                "status": solution['status'],
                "objective_value": solution['objective_value'],
                "selected_facilities": solution['selected_facilities'],
                "assignments": solution.get('assignments', {}),
                "metrics": metrics,
                "solution_time": solution_time,
                "solver_details": solution.get('solver_details', {}),
                "variant_used": variant,
                "warnings": solution.get('warnings', []),
                "service_radius_unit": service_radius_unit,  # Pass unit for visualization
                "academic_metadata": {
                    "algorithm_used": "Mixed Integer Programming (Set Cover)",
                    "references": self.get_metadata()['academic_refs'][:2],
                    "assumptions": [
                        f"Service radius: {service_radius} {service_radius_unit}",
                        "All demand must be covered",
                        f"Distance metric: {distance_metric}",
                        "Facilities have unlimited capacity"
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error solving LSCP: {e}")
            return {
                "status": "error",
                "error": str(e),
                "solution_time": time.time() - start_time
            }
    
    def _solve_mip(
        self,
        coverage_matrix: np.ndarray,
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float] = None,
        variant: str = "base",
        k: int = 2,
        weights: Optional[np.ndarray] = None,
        reliability: Optional[np.ndarray] = None,
        alpha: float = 0.95,
        fraction: float = 0.95,
        existing: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            return self._solve_gurobi(
                coverage_matrix, constraints, time_limit_seconds,
                variant=variant, k=k, weights=weights, reliability=reliability,
                alpha=alpha, fraction=fraction, existing=existing
            )
        except ImportError:
            logger.info("Gurobi not available, using PuLP")
            return self._solve_pulp(
                coverage_matrix, constraints, time_limit_seconds,
                variant=variant, k=k, weights=weights, reliability=reliability,
                alpha=alpha, fraction=fraction, existing=existing
            )
        except gp.GurobiError as e:
            logger.warning("Gurobi unavailable (%s), falling back to PuLP", e)
            return self._solve_pulp(
                coverage_matrix, constraints, time_limit_seconds,
                variant=variant, k=k, weights=weights, reliability=reliability,
                alpha=alpha, fraction=fraction, existing=existing
            )
    
    def _solve_gurobi(
        self,
        coverage_matrix: np.ndarray,
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float] = None,
        variant: str = "base",
        k: int = 2,
        weights: Optional[np.ndarray] = None,
        reliability: Optional[np.ndarray] = None,
        alpha: float = 0.95,
        fraction: float = 0.95,
        existing: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        import gurobipy as gp
        from gurobipy import GRB

        from .base_solver import configure_gurobi_model

        n_demand, n_candidates = coverage_matrix.shape
        existing_set = set(existing or [])

        model = gp.Model("lscp")
        configure_gurobi_model(model, time_limit_seconds)

        # Decision variables
        x = model.addVars(n_candidates, vtype=GRB.BINARY, name="x")

        # Objective
        if variant == "conditional":
            # Minimize ADDITIONAL facilities (pre-existing ones are forced open)
            model.setObjective(
                gp.quicksum(x[j] for j in range(n_candidates) if j not in existing_set),
                GRB.MINIMIZE
            )
        else:
            # Minimize total number of facilities
            model.setObjective(gp.quicksum(x[j] for j in range(n_candidates)), GRB.MINIMIZE)

        # Coverage indicator variables for the partial variant
        z = None
        if variant == "partial":
            z = model.addVars(n_demand, vtype=GRB.BINARY, name="z")

        # Coverage constraints
        for i in range(n_demand):
            covering_facilities = [j for j in range(n_candidates) if coverage_matrix[i, j] == 1]
            if variant == "backup":
                # Each demand must be covered by at least k facilities
                model.addConstr(
                    gp.quicksum(x[j] for j in covering_facilities) >= k,
                    f"cover_{i}"
                )
            elif variant == "probabilistic":
                # Reliability-based redundancy: sum of -ln(1-p_j) over covering facilities >= -ln(1-alpha)
                coeffs = {
                    j: -math.log(1.0 - min(float(reliability[j]), 1.0 - 1e-9))
                    for j in covering_facilities
                }
                if covering_facilities:
                    model.addConstr(
                        gp.quicksum(coeffs[j] * x[j] for j in covering_facilities) >= -math.log(1.0 - alpha),
                        f"cover_{i}"
                    )
                else:
                    # This should have been caught in feasibility check
                    raise ValueError(f"Demand point {i} cannot be covered")
            elif variant == "partial":
                # z[i] can only be 1 if at least one covering facility is selected
                if covering_facilities:
                    model.addConstr(
                        z[i] <= gp.quicksum(x[j] for j in covering_facilities),
                        f"cover_{i}"
                    )
                else:
                    model.addConstr(z[i] == 0, f"cover_{i}")
            else:
                # base / conditional: each demand covered by at least one facility
                if covering_facilities:
                    model.addConstr(
                        gp.quicksum(x[j] for j in covering_facilities) >= 1,
                        f"cover_{i}"
                    )
                else:
                    # This should have been caught in feasibility check
                    raise ValueError(f"Demand point {i} cannot be covered")

        # Conditional variant: force pre-existing facilities open
        if variant == "conditional":
            for j in existing_set:
                if 0 <= j < n_candidates:
                    model.addConstr(x[j] == 1)

        # Partial variant: cover at least a fraction of total weighted demand
        if variant == "partial":
            w = weights if weights is not None else np.ones(n_demand)
            total_weight = float(np.sum(w))
            model.addConstr(
                gp.quicksum(float(w[i]) * z[i] for i in range(n_demand)) >= fraction * total_weight,
                "partial_coverage"
            )

        # Custom constraints
        must_include = constraints.get('must_include', [])
        for j in must_include:
            if 0 <= j < n_candidates:
                model.addConstr(x[j] == 1)

        must_exclude = constraints.get('must_exclude', [])
        for j in must_exclude:
            if 0 <= j < n_candidates:
                model.addConstr(x[j] == 0)

        # Maximum budget constraint (if provided)
        max_facilities = constraints.get('max_facilities')
        if max_facilities:
            model.addConstr(gp.quicksum(x[j] for j in range(n_candidates)) <= max_facilities)

        model.optimize()

        timed_out = (model.status == GRB.TIME_LIMIT)
        if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            selected = [j for j in range(n_candidates) if x[j].X > 0.5]

            # Determine assignments
            assignments = {}
            for i in range(n_demand):
                if variant == "partial" and z[i].X <= 0.5:
                    # Demand intentionally left uncovered under the partial variant
                    continue
                # Assign to nearest selected facility that covers this demand
                covering_selected = [j for j in selected if coverage_matrix[i, j] == 1]
                if covering_selected:
                    assignments[i] = covering_selected[0]  # Just pick first one

            return {
                'status': 'optimal' if model.status == GRB.OPTIMAL else 'feasible',
                'objective_value': len(selected),
                'selected_facilities': selected,
                'assignments': assignments,
                'solver_details': {
                    'solver': 'gurobi',
                    'gap': model.MIPGap,
                    'formulation': f'LSCP Set Cover MIP ({variant})',
                    'timed_out': bool(timed_out)
                }
            }
        else:
            return {
                'status': 'infeasible',
                'objective_value': None,
                'selected_facilities': [],
                'assignments': {}
            }
    
    def _solve_pulp(
        self,
        coverage_matrix: np.ndarray,
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float] = None,
        variant: str = "base",
        k: int = 2,
        weights: Optional[np.ndarray] = None,
        reliability: Optional[np.ndarray] = None,
        alpha: float = 0.95,
        fraction: float = 0.95,
        existing: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        import pulp

        n_demand, n_candidates = coverage_matrix.shape
        existing_set = set(existing or [])

        prob = pulp.LpProblem("lscp", pulp.LpMinimize)

        x = pulp.LpVariable.dicts("x", range(n_candidates), cat='Binary')

        # Coverage indicator variables for the partial variant
        z = None
        if variant == "partial":
            z = pulp.LpVariable.dicts("z", range(n_demand), cat='Binary')

        # Objective
        if variant == "conditional":
            # Minimize ADDITIONAL facilities (pre-existing ones are forced open)
            prob += pulp.lpSum([x[j] for j in range(n_candidates) if j not in existing_set])
        else:
            prob += pulp.lpSum([x[j] for j in range(n_candidates)])

        # Coverage constraints
        for i in range(n_demand):
            covering_facilities = [j for j in range(n_candidates) if coverage_matrix[i, j] == 1]
            if variant == "backup":
                prob += pulp.lpSum([x[j] for j in covering_facilities]) >= k
            elif variant == "probabilistic":
                if covering_facilities:
                    coeffs = {
                        j: -math.log(1.0 - min(float(reliability[j]), 1.0 - 1e-9))
                        for j in covering_facilities
                    }
                    prob += pulp.lpSum([coeffs[j] * x[j] for j in covering_facilities]) >= -math.log(1.0 - alpha)
                else:
                    raise ValueError(f"Demand point {i} cannot be covered")
            elif variant == "partial":
                if covering_facilities:
                    prob += z[i] <= pulp.lpSum([x[j] for j in covering_facilities])
                else:
                    prob += z[i] == 0
            else:
                if covering_facilities:
                    prob += pulp.lpSum([x[j] for j in covering_facilities]) >= 1
                else:
                    raise ValueError(f"Demand point {i} cannot be covered")

        # Conditional variant: force pre-existing facilities open
        if variant == "conditional":
            for j in existing_set:
                if 0 <= j < n_candidates:
                    prob += x[j] == 1

        # Partial variant: cover at least a fraction of total weighted demand
        if variant == "partial":
            w = weights if weights is not None else np.ones(n_demand)
            total_weight = float(np.sum(w))
            prob += pulp.lpSum([float(w[i]) * z[i] for i in range(n_demand)]) >= fraction * total_weight

        must_include = constraints.get('must_include', [])
        for j in must_include:
            if 0 <= j < n_candidates:
                prob += x[j] == 1

        must_exclude = constraints.get('must_exclude', [])
        for j in must_exclude:
            if 0 <= j < n_candidates:
                prob += x[j] == 0

        max_facilities = constraints.get('max_facilities')
        if max_facilities:
            prob += pulp.lpSum([x[j] for j in range(n_candidates)]) <= max_facilities

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=float(time_limit_seconds) if time_limit_seconds is not None else None)
        prob.solve(solver)
        timed_out = bool(time_limit_seconds is not None and prob.status not in (pulp.LpStatusOptimal, pulp.LpStatusInfeasible))

        if prob.status == pulp.LpStatusOptimal:
            selected = [j for j in range(n_candidates) if pulp.value(x[j]) > 0.5]

            assignments = {}
            for i in range(n_demand):
                if variant == "partial" and (pulp.value(z[i]) or 0) <= 0.5:
                    # Demand intentionally left uncovered under the partial variant
                    continue
                covering_selected = [j for j in selected if coverage_matrix[i, j] == 1]
                if covering_selected:
                    assignments[i] = covering_selected[0]

            return {
                'status': 'optimal',
                'objective_value': len(selected),
                'selected_facilities': selected,
                'assignments': assignments,
                'solver_details': {
                    'solver': 'pulp',
                    'formulation': f'LSCP Set Cover MIP ({variant})',
                    'timed_out': timed_out
                }
            }
        else:
            return {
                'status': 'infeasible',
                'objective_value': None,
                'selected_facilities': [],
                'assignments': {},
                'solver_details': {'solver': 'pulp', 'timed_out': timed_out}
            }
    
    def _calculate_metrics(
        self,
        coverage_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        selected_facilities: List[int],
        service_radius: float,
        service_radius_unit: Optional[str] = None
    ) -> Dict[str, float]:
        n_demand = coverage_matrix.shape[0]
        num_facilities = len(selected_facilities)
        
        # All demands should be covered
        covered = np.zeros(n_demand, dtype=bool)
        for j in selected_facilities:
            covered |= (coverage_matrix[:, j] == 1)
        
        coverage_pct = (np.sum(covered) / n_demand * 100) if n_demand > 0 else 0
        
        # LSCP validation: coverage should be 100% for a valid solution
        if coverage_pct < 100.0 and num_facilities > 0:
            logger.warning(
                f"LSCP solution has {coverage_pct:.1f}% coverage instead of 100%. "
                f"This indicates a potential issue with the solution or data."
            )
        
        if num_facilities > 0:
            min_distances = np.min(distance_matrix[:, selected_facilities], axis=1)
            avg_distance = float(np.mean(min_distances))
            max_distance = float(np.max(min_distances))
        else:
            avg_distance = 0.0
            max_distance = 0.0
            
        # Convert distances to user-requested units if specified
        from utils.distance_calculator import DistanceCalculator
        dist_calc = DistanceCalculator()
        
        return {
            # Objective info for LSCP (minimize facilities)
            "objective_value": float(num_facilities),
            "objective_name": "min_facilities",
            # Core metrics
            "num_facilities": num_facilities,
            "coverage_percentage": float(coverage_pct),
            "num_covered_points": int(np.sum(covered)),
            "num_uncovered_points": int(n_demand - np.sum(covered)),
            "service_radius": float(service_radius),
            "average_distance": dist_calc.convert_meters_to_unit(avg_distance, service_radius_unit),
            "max_distance": dist_calc.convert_meters_to_unit(max_distance, service_radius_unit),
            "total_demand_points": n_demand
        }
    
    def explain_solution(
        self,
        solution: Dict[str, Any],
        data: Dict[str, gpd.GeoDataFrame],
        detail_level: str = "standard"
    ) -> str:
        if solution.get('status') == 'error':
            return f"❌ Solution failed: {solution.get('error', 'Unknown error')}"
        
        if solution.get('status') == 'infeasible':
            error_msg = solution.get('error', 'No feasible solution found.')
            return f"❌ {error_msg}"
        
        metrics = solution.get('metrics', {})
        n_facilities = metrics.get('num_facilities', 0)
        service_radius = metrics.get('service_radius', 0)
        coverage_pct = metrics.get('coverage_percentage', 0)
        
        if detail_level == "brief":
            return f"Minimum {n_facilities} facilities needed for full coverage within {service_radius:.2f}."
        
        else:
            return f"""
**LSCP Solution Summary**

✅ Successfully minimized facility count while ensuring full coverage.

**Solution:**
- **Minimum Facilities Required:** {n_facilities}
- **Service Radius:** {service_radius:.2f}
- **Coverage:** {coverage_pct:.1f}%

**Performance:**
- Total Demand Points: {metrics.get('total_demand_points', 0)}
- Covered Points: {metrics.get('num_covered_points', 0)}
- Average Distance: {metrics.get('average_distance', 0):.2f}
- Maximum Distance: {metrics.get('max_distance', 0):.2f}

This solution uses the minimum number of facilities to ensure all demand points are within the service radius.
            """.strip()
    
    def get_visualization_config(self) -> Dict[str, Any]:
        config = super().get_visualization_config()
        config['show_service_areas'] = True
        return config

