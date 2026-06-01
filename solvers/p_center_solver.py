from .base_solver import SpatialOptimizationProblem
from typing import Dict, List, Any, Optional
import geopandas as gpd
import numpy as np
import time
import logging

from utils.heuristics.genetic_solver import GAConfig, PCenterGeneticSolver

logger = logging.getLogger(__name__)

class PCenterSolver(SpatialOptimizationProblem):
    """
    P-Center Problem Solver
    
    Minimizes the maximum distance from any demand point to its nearest facility (minimax objective).
    """
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "P-Center Problem",
            "short_name": "p-center",
            "category": "equity/minimax",
            "description": "Locate p facilities to minimize the maximum distance from any demand point to its nearest facility. Ensures equitable service by minimizing worst-case access distance.",
            "mathematical_formulation": """
Minimize: W (maximum service distance)

Subject to:
- Σⱼ xⱼ = p  (locate exactly p facilities)
- Σⱼ yᵢⱼ = 1, ∀i  (each demand assigned to one facility)
- yᵢⱼ ≤ xⱼ, ∀i,j  (assignment only to open facilities)
- W ≥ dᵢⱼ · yᵢⱼ, ∀i,j  (W is maximum distance)
- xⱼ, yᵢⱼ ∈ {0,1}

Where:
- W = maximum service distance (minimax objective)
- dᵢⱼ = distance from demand i to candidate site j
- xⱼ = 1 if facility located at j
- yᵢⱼ = 1 if demand i assigned to facility j
- p = number of facilities
            """,
            "academic_refs": [
                "Hakimi, S. L. (1964). Optimum locations of switching centers and the absolute centers and medians of a graph. Operations Research, 12(3), 450-459.",
                "Daskin, M. S. (1995). Network and discrete location: models, algorithms, and applications. John Wiley & Sons.",
                "Minieka, E. (1970). The m-center problem. SIAM Review, 12(1), 138-139."
            ],
            "complexity": "NP-hard",
            "typical_use_cases": [
                "Emergency service location (ambulance, fire stations)",
                "Disaster relief facility placement",
                "Equal access public services",
                "Strategic facility placement for equity",
                "Maximum coverage with fairness constraint"
            ],
            "keywords": [
                "p-center", "minimax", "minimize maximum distance",
                "worst case", "equity", "fairness", "maximum distance",
                "emergency", "equal access"
            ],
            "variants": ["vertex", "weighted", "conditional"]
        }
    
    def get_conversation_prompts(self) -> Dict[str, Any]:
        return {
            "problem_detection": [
                "minimax", "minimize maximum", "worst case", "maximum distance",
                "equity", "fairness", "emergency", "p-center"
            ],
            "parameter_questions": [
                {
                    "param": "n_facilities",
                    "question": "How many facilities would you like to locate?",
                    "type": "int",
                    "validation": "Must be a positive integer",
                    "help": "The number of facilities (p) to establish"
                }
            ],
            "constraint_suggestions": [
                "Would you like to specify any facilities that must be included?",
                "Are there any candidate sites that should be excluded?",
                "Do you have a target maximum distance threshold?"
            ],
            "explanation_template": "The P-Center solution locates {n_facilities} facilities to minimize maximum distance. Worst-case distance: {max_dist:.2f}."
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
        if "n_facilities" not in params:
            return False, "Missing required parameter: n_facilities (p)"
        
        n_facilities = params["n_facilities"]
        if not isinstance(n_facilities, int) or n_facilities <= 0:
            return False, "n_facilities must be a positive integer"

        ok, err = self._validate_facility_sets(params)
        if not ok:
            return False, err

        variant = params.get("variant", "vertex")
        allowed = {"vertex", "weighted", "conditional"}
        if variant not in allowed:
            return False, f"Unknown P-Center variant '{variant}'. Allowed variants: {sorted(allowed)}"

        if variant == "conditional":
            existing = params.get("existing_facilities")
            if not existing:
                return False, "Conditional P-Center requires a non-empty 'existing_facilities' list (candidate-site indices)"

        return True, None

    def _extract_weights(self, demand_gdf: gpd.GeoDataFrame, parameters: Dict[str, Any]) -> np.ndarray:
        """Extract demand weights from GeoDataFrame, preferring population columns and explicit parameter."""
        # 1) Explicit parameter takes precedence
        try:
            explicit_col = parameters.get('demand_weight_column') if parameters else None
            if explicit_col:
                for c in demand_gdf.columns:
                    if c.lower() == str(explicit_col).lower():
                        values = demand_gdf[c].astype(float).to_numpy()
                        if np.any(values < 0):
                            raise ValueError(f"Demand weight column '{c}' contains negative values")
                        return values
        except Exception as e:
            logger.warning(f"Failed to use explicit demand_weight_column: {e}")

        # 2) Case-insensitive exact matches of common names
        common_exact = ['population', 'pop', 'demand', 'weight']
        lower_cols = {c.lower(): c for c in demand_gdf.columns}
        for key in common_exact:
            if key in lower_cols:
                c = lower_cols[key]
                try:
                    values = demand_gdf[c].astype(float).to_numpy()
                    if np.all(values >= 0):
                        return values
                except Exception:
                    pass

        # 3) Substring heuristic
        substr_keys = ['population', 'pop', 'weight', 'demand']
        for c in demand_gdf.columns:
            lc = c.lower()
            if any(k in lc for k in substr_keys):
                try:
                    values = demand_gdf[c].astype(float).to_numpy()
                    if np.all(values >= 0):
                        return values
                except Exception:
                    continue

        logger.info("No weight column found, using uniform weights of 1.0")
        return np.ones(len(demand_gdf))

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
            
            p = parameters['n_facilities']
            service_radius_unit = parameters.get('service_radius_unit', 'm')
            variant = parameters.get('variant', 'vertex')

            weights = self._extract_weights(demand_gdf, parameters) if variant == 'weighted' else None

            existing = None
            if variant == 'conditional':
                existing = sorted(set(int(v) for v in (parameters.get('existing_facilities') or [])))

            constraints = self._merge_facility_set_constraints(constraints, parameters)

            if p > len(candidate_gdf):
                raise ValueError(f"Cannot locate {p} facilities with only {len(candidate_gdf)} candidate sites")

            if variant == 'conditional' and p > len(candidate_gdf) - len(existing):
                raise ValueError(
                    f"Cannot locate {p} new facilities: only {len(candidate_gdf) - len(existing)} "
                    f"candidate sites remain after {len(existing)} existing facilities are fixed open"
                )

            # Calculate distance matrix
            dist_calc = DistanceCalculator()
            network_graph = data.get('_network_graph')
            distance_matrix = dist_calc.calculate_distance_matrix(
                demand_gdf, candidate_gdf, metric=distance_metric, network_graph=network_graph
            )
            
            fallback_time_limit = float(parameters.get('fallback_time_limit_seconds', 60.0))
            ga_time_budget = float(parameters.get('ga_time_budget_seconds', 60.0))
            logger.info(f"P-Center: Fallback time limit set to {fallback_time_limit:.2f} seconds, GA time budget: {ga_time_budget:.2f} seconds")
            
            # Solve using MIP with a strict time limit
            mip_start = time.time()
            solution = self._solve_mip(
                distance_matrix,
                p,
                constraints,
                time_limit_seconds=fallback_time_limit,
                variant=variant,
                weights=weights,
                existing=existing
            )
            mip_elapsed = time.time() - mip_start

            timed_out_flag = bool(solution.get('solver_details', {}).get('timed_out', False))
            ga_needed = timed_out_flag or (
                fallback_time_limit > 0 and mip_elapsed >= max(0.1, 0.95 * fallback_time_limit)
            )
            logger.info(f"P-Center timeout check: mip_elapsed={mip_elapsed:.2f}s, fallback_limit={fallback_time_limit:.2f}s, timed_out_flag={timed_out_flag}, ga_needed={ga_needed}")
            if ga_needed:
                ga_cfg = GAConfig(time_limit_seconds=ga_time_budget)
                ga_solver = PCenterGeneticSolver(ga_cfg)
                if ga_solver.supports_variant(variant):
                    logger.info("P-Center: Falling back to Genetic Algorithm")
                    logger.info(f"P-Center: MIP solver status: {solution.get('status', 'unknown')}, objective: {solution.get('objective_value', 'N/A')}")
                    incumbent_mask = None
                    if solution.get('selected_facilities'):
                        incumbent_mask = np.zeros(distance_matrix.shape[1], dtype=np.int8)
                        for idx in solution['selected_facilities']:
                            if 0 <= idx < incumbent_mask.size:
                                incumbent_mask[idx] = 1
                    logger.info(f"P-Center: Starting GA with time budget: {ga_time_budget:.2f} seconds")
                    ga_result = ga_solver.solve(
                        distance_matrix=distance_matrix,
                        p=p,
                        initial_solution=incumbent_mask,
                        time_budget_seconds=ga_time_budget,
                        variant=variant,
                        demand_weights=weights,
                        existing=existing,
                    )
                    logger.info(f"P-Center: GA completed with status: {ga_result.get('status', 'unknown')}, objective: {ga_result.get('objective_value', 'N/A')}")
                    ga_details = {
                        **ga_result.get('solver_details', {}),
                        "fallback_from": solution.get('solver_details', {}).get('solver', 'mip'),
                        "fallback_reason": "time_limit"
                    }
                    ga_warnings = []
                    if ga_result.get("status") == "approximate":
                        ga_warnings.append(
                            f"GA fallback returned an approximate solution that may marginally "
                            f"violate the '{variant}' constraint."
                        )
                    solution = {
                        "status": ga_result.get("status", "feasible"),
                        "objective_value": ga_result["objective_value"],
                        "selected_facilities": ga_result["selected_facilities"],
                        "assignments": ga_result["assignments"],
                        "solver_details": ga_details,
                        "warnings": ga_warnings,
                    }
                else:
                    logger.info(f"P-Center: GA fallback not available for variant '{variant}'; keeping MIP incumbent")
                    if timed_out_flag:
                        solution.setdefault('warnings', []).append(
                            f"MIP time limit reached for the '{variant}' variant; returning best incumbent "
                            f"(genetic-algorithm fallback is not available for this variant)."
                        )
            else:
                logger.info(f"P-Center: MIP solver completed successfully within time limit, no fallback needed. Status: {solution.get('status', 'unknown')}, objective: {solution.get('objective_value', 'N/A')}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                distance_matrix, 
                solution['selected_facilities'],
                solution['assignments'],
                distance_unit=service_radius_unit
            )
            
            solution_time = time.time() - start_time

            objective_name = "weighted_max_distance" if variant == "weighted" else "max_distance"

            result = {
                "status": solution['status'],
                "objective_value": solution['objective_value'],
                "selected_facilities": solution['selected_facilities'],
                "assignments": solution['assignments'],
                "variant_used": variant,
                "metrics": {
                    **metrics,
                    "objective_value": float(solution['objective_value']),
                    "objective_name": objective_name
                },
                "solution_time": solution_time,
                "solver_details": solution.get('solver_details', {}),
                "academic_metadata": {
                    "algorithm_used": "Mixed Integer Programming (MIP) - Minimax formulation",
                    "references": self.get_metadata()['academic_refs'][:2],
                    "assumptions": [
                        "Each demand point is assigned to exactly one facility",
                        "Facilities have unlimited capacity",
                        f"Distance metric: {distance_metric}",
                        "Objective: minimize maximum service distance"
                    ]
                }
            }

            if solution.get('warnings'):
                result.setdefault('warnings', []).extend(solution['warnings'])

            return result

        except Exception as e:
            logger.error(f"Error solving P-Center problem: {e}")
            return {
                "status": "error",
                "error": str(e),
                "solution_time": time.time() - start_time
            }
    
    def _solve_mip(
        self,
        distance_matrix: np.ndarray,
        p: int,
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float] = None,
        variant: str = "vertex",
        weights: Optional[np.ndarray] = None,
        existing: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Solve using MIP"""
        try:
            import gurobipy as gp
            from gurobipy import GRB
            return self._solve_gurobi(distance_matrix, p, constraints, time_limit_seconds, variant=variant, weights=weights, existing=existing)
        except ImportError:
            logger.info("Gurobi not available, using PuLP")
            return self._solve_pulp(distance_matrix, p, constraints, time_limit_seconds, variant=variant, weights=weights, existing=existing)
    
    def _solve_gurobi(
        self,
        distance_matrix: np.ndarray,
        p: int,
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float] = None,
        variant: str = "vertex",
        weights: Optional[np.ndarray] = None,
        existing: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        import gurobipy as gp
        from gurobipy import GRB

        from .base_solver import configure_gurobi_model

        n_demand, n_candidates = distance_matrix.shape
        existing_set = set(existing or [])

        model = gp.Model("p-center")
        configure_gurobi_model(model, time_limit_seconds)

        x = model.addVars(n_candidates, vtype=GRB.BINARY, name="x")
        y = model.addVars(n_demand, n_candidates, vtype=GRB.BINARY, name="y")
        W = model.addVar(vtype=GRB.CONTINUOUS, name="W")  # Maximum distance

        model.setObjective(W, GRB.MINIMIZE)

        if variant == "conditional":
            # Existing facilities are forced open and do not consume the budget;
            # p counts only the new facilities placed among the remaining candidates.
            for j in existing_set:
                if 0 <= j < n_candidates:
                    model.addConstr(x[j] == 1)
            model.addConstr(
                gp.quicksum(x[j] for j in range(n_candidates) if j not in existing_set) == p
            )
        else:
            model.addConstr(gp.quicksum(x[j] for j in range(n_candidates)) == p)

        model.addConstrs(
            (gp.quicksum(y[i, j] for j in range(n_candidates)) == 1
             for i in range(n_demand)),
            name="assign_demand",
        )

        # Batched linking + Chebyshev constraints (Gurobi batches much faster
        # than a Python double loop with individual addConstr calls).
        model.addConstrs(
            (y[i, j] <= x[j] for i in range(n_demand) for j in range(n_candidates)),
            name="y_le_x",
        )
        if variant == "weighted":
            model.addConstrs(
                (W >= float(weights[i]) * float(distance_matrix[i, j]) * y[i, j]
                 for i in range(n_demand) for j in range(n_candidates)),
                name="radius_cover",
            )
        else:
            model.addConstrs(
                (W >= float(distance_matrix[i, j]) * y[i, j]
                 for i in range(n_demand) for j in range(n_candidates)),
                name="radius_cover",
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
        
        model.optimize()
        
        timed_out = (model.status == GRB.TIME_LIMIT)
        if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            selected = [j for j in range(n_candidates) if x[j].X > 0.5]
            assignments = {}
            for i in range(n_demand):
                for j in range(n_candidates):
                    if y[i, j].X > 0.5:
                        assignments[i] = j
                        break
            
            return {
                'status': 'optimal' if model.status == GRB.OPTIMAL else 'feasible',
                'objective_value': W.X,
                'selected_facilities': selected,
                'assignments': assignments,
                'solver_details': {
                    'solver': 'gurobi',
                    'gap': model.MIPGap,
                    'formulation': 'P-Center Minimax MIP',
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
        distance_matrix: np.ndarray,
        p: int,
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float] = None,
        variant: str = "vertex",
        weights: Optional[np.ndarray] = None,
        existing: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        import pulp

        n_demand, n_candidates = distance_matrix.shape
        existing_set = set(existing or [])

        prob = pulp.LpProblem("p-center", pulp.LpMinimize)

        x = pulp.LpVariable.dicts("x", range(n_candidates), cat='Binary')
        y = pulp.LpVariable.dicts("y",
            ((i, j) for i in range(n_demand) for j in range(n_candidates)),
            cat='Binary')
        W = pulp.LpVariable("W", lowBound=0)

        prob += W

        if variant == "conditional":
            # Existing facilities are forced open and do not consume the budget;
            # p counts only the new facilities placed among the remaining candidates.
            for j in existing_set:
                if 0 <= j < n_candidates:
                    prob += x[j] == 1
            prob += pulp.lpSum([x[j] for j in range(n_candidates) if j not in existing_set]) == p
        else:
            prob += pulp.lpSum([x[j] for j in range(n_candidates)]) == p

        for i in range(n_demand):
            prob += pulp.lpSum([y[(i, j)] for j in range(n_candidates)]) == 1

        for i in range(n_demand):
            for j in range(n_candidates):
                prob += y[(i, j)] <= x[j]
                if variant == "weighted":
                    prob += W >= float(weights[i]) * distance_matrix[i, j] * y[(i, j)]
                else:
                    prob += W >= distance_matrix[i, j] * y[(i, j)]

        must_include = constraints.get('must_include', [])
        for j in must_include:
            if 0 <= j < n_candidates:
                prob += x[j] == 1
        
        must_exclude = constraints.get('must_exclude', [])
        for j in must_exclude:
            if 0 <= j < n_candidates:
                prob += x[j] == 0
        
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=float(time_limit_seconds) if time_limit_seconds is not None else None)
        prob.solve(solver)
        timed_out = bool(time_limit_seconds is not None and prob.status not in (pulp.LpStatusOptimal, pulp.LpStatusInfeasible))
        
        if prob.status == pulp.LpStatusOptimal:
            selected = [j for j in range(n_candidates) if pulp.value(x[j]) > 0.5]
            assignments = {}
            for i in range(n_demand):
                for j in range(n_candidates):
                    if pulp.value(y[(i, j)]) > 0.5:
                        assignments[i] = j
                        break
            
            return {
                'status': 'optimal',
                'objective_value': pulp.value(W),
                'selected_facilities': selected,
                'assignments': assignments,
                'solver_details': {
                    'solver': 'pulp',
                    'formulation': 'P-Center Minimax MIP',
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
        distance_matrix: np.ndarray,
        selected_facilities: List[int],
        assignments: Dict[int, int],
        distance_unit: Optional[str] = None
    ) -> Dict[str, float]:
        distances = [distance_matrix[i, j] for i, j in assignments.items()]
        
        # Convert distances to user-requested units if specified
        from utils.distance_calculator import DistanceCalculator
        dist_calc = DistanceCalculator()
        
        max_dist = max(distances) if distances else 0
        avg_dist = np.mean(distances) if distances else 0
        min_dist = min(distances) if distances else 0
        std_dist = np.std(distances) if distances else 0
        
        return {
            "max_distance": dist_calc.convert_meters_to_unit(float(max_dist), distance_unit),
            "average_distance": dist_calc.convert_meters_to_unit(float(avg_dist), distance_unit),
            "min_distance": dist_calc.convert_meters_to_unit(float(min_dist), distance_unit),
            "std_distance": dist_calc.convert_meters_to_unit(float(std_dist), distance_unit),
            "num_facilities": len(selected_facilities),
            "num_demand_points": len(assignments)
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
            return "❌ No feasible solution found."
        
        metrics = solution.get('metrics', {})
        n_facilities = metrics.get('num_facilities', 0)
        max_dist = metrics.get('max_distance', 0)
        avg_dist = metrics.get('average_distance', 0)
        
        if detail_level == "brief":
            return f"Located {n_facilities} facilities with maximum distance {max_dist:.2f}."
        
        elif detail_level == "standard":
            return f"""
**P-Center Solution Summary**

✅ Successfully located {n_facilities} facilities to minimize maximum service distance.

**Key Metrics:**
- Maximum Distance (worst case): {max_dist:.2f}
- Average Distance: {avg_dist:.2f}
- Minimum Distance (best case): {metrics.get('min_distance', 0):.2f}

The solution ensures equitable access by minimizing the worst-case distance. No demand point is farther than {max_dist:.2f} from its assigned facility.
            """.strip()
        
        else:
            return f"Located {n_facilities} facilities with maximum service distance of {max_dist:.2f}."
    
    def get_visualization_config(self) -> Dict[str, Any]:
        config = super().get_visualization_config()
        config['show_service_areas'] = True  # Show service areas for P-Center
        return config

