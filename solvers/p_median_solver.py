from .base_solver import SpatialOptimizationProblem
from typing import Dict, List, Any, Optional
import geopandas as gpd
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class PMedianSolver(SpatialOptimizationProblem):
    """
    P-Median Problem Solver
    
    Minimizes the total (or average) weighted distance from demand points to facilities.
    """
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "P-Median Problem",
            "short_name": "p-median",
            "category": "distance minimization",
            "description": "Locate p facilities to minimize the total or average weighted distance from demand points to their nearest facility. Optimal for minimizing access costs or travel distances.",
            "mathematical_formulation": """
Minimize: Σᵢ Σⱼ dᵢⱼ · wᵢ · yᵢⱼ

Subject to:
- Σⱼ xⱼ = p  (locate exactly p facilities)
- Σⱼ yᵢⱼ = 1, ∀i  (each demand assigned to one facility)
- yᵢⱼ ≤ xⱼ, ∀i,j  (assignment only to open facilities)
- xⱼ, yᵢⱼ ∈ {0,1}

Where:
- dᵢⱼ = distance from demand i to candidate site j
- wᵢ = weight (demand) at point i
- xⱼ = 1 if facility located at j, 0 otherwise
- yᵢⱼ = 1 if demand i assigned to facility j, 0 otherwise
- p = number of facilities to locate
            """,
            "academic_refs": [
                "Hakimi, S. L. (1964). Optimum locations of switching centers and the absolute centers and medians of a graph. Operations Research, 12(3), 450-459.",
                "ReVelle, C. S., & Swain, R. W. (1970). Central facilities location. Geographical Analysis, 2(1), 30-42.",
                "Kariv, O., & Hakimi, S. L. (1979). An algorithmic approach to network location problems. SIAM Journal on Applied Mathematics, 37(3), 513-538.",
                "Daskin, M. S. (2013). Network and discrete location: models, algorithms, and applications. John Wiley & Sons."
            ],
            "complexity": "NP-hard",
            "typical_use_cases": [
                "Warehouse location to minimize distribution costs",
                "Public facility siting (libraries, schools, post offices)",
                "Emergency service station location",
                "Retail store placement",
                "Distribution center optimization"
            ],
            "keywords": [
                "p-median", "minimize distance", "minimize average distance", "average distance",
                "minimize total distance", "minimize cost", "access optimization",
                "facility location", "median", "distribution"
            ],
            "variants": [
                "base",
                "capacitated",
                "budget",
                "max_distance"
            ],
            "parameters": {
                "n_facilities": {"type": "int", "required": True},
                "objective": {"type": "choice", "choices": ["total", "average"], "required": False},
                "variant": {"type": "choice", "choices": ["base", "capacitated", "budget", "max_distance"], "required": False},
                "capacities": {"type": "list[float]", "required": False},
                "facility_costs": {"type": "list[float]", "required": False},
                "budget": {"type": "float", "required": False},
                "max_assignment_distance": {"type": "float", "required": False},
                "service_radius": {"type": "float", "required": False, "description": "Maximum service radius for assignment validation"}
            }
        }
    
    def get_conversation_prompts(self) -> Dict[str, Any]:
        return {
            "problem_detection": [
                "minimize distance", "minimize average", "minimize total",
                "minimize cost", "access", "p-median", "median"
            ],
            "parameter_questions": [
                {
                    "param": "n_facilities",
                    "question": "How many facilities would you like to locate?",
                    "type": "int",
                    "validation": "Must be a positive integer",
                    "help": "The number of facilities (p) to establish"
                },
                {
                    "param": "objective",
                    "question": "Would you like to minimize 'total' distance or 'average' distance?",
                    "type": "choice",
                    "choices": ["total", "average"],
                    "default": "total",
                    "help": "Total weights all demands equally, average normalizes by total demand"
                }
            ],
            "constraint_suggestions": [
                "Would you like to specify any facilities that must be included?",
                "Are there any candidate sites that should be excluded?",
                "Do you want to set a maximum distance threshold?",
                "Do you want to validate assignments against a service radius?"
            ],
            "explanation_template": "The P-Median solution locates {n_facilities} facilities to minimize the {objective} weighted distance. Total objective value: {obj_value:.2f}. Average distance: {avg_dist:.2f}."
        }
    
    def get_required_data(self) -> Dict[str, Dict[str, Any]]:
        return {
            "demand_points": {
                "required": True,
                "description": "Points representing demand locations (e.g., population centers, customers)",
                "required_fields": [],
                "optional_fields": ["demand", "weight", "population"],
                "geometry_type": "Point"
            },
            "candidate_sites": {
                "required": True,
                "description": "Potential facility locations to choose from",
                "required_fields": [],
                "optional_fields": ["capacity", "cost"],
                "geometry_type": "Point"
            }
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate problem-specific parameters"""
        if "n_facilities" not in params:
            return False, "Missing required parameter: n_facilities (p)"
        
        n_facilities = params["n_facilities"]
        
        if not isinstance(n_facilities, int) or n_facilities <= 0:
            return False, "n_facilities must be a positive integer"
        
        # Check objective if provided
        if "objective" in params:
            if params["objective"] not in ["total", "average"]:
                return False, "objective must be either 'total' or 'average'"
        
        # Cross-cutting facility-set params
        ok, err = self._validate_facility_sets(params)
        if not ok:
            return False, err

        # Variant validation
        variant = params.get("variant", "base")
        if variant not in ["base", "capacitated", "budget", "max_distance"]:
            return False, "variant must be one of: base, capacitated, budget, max_distance"
        if variant == "budget":
            if "budget" not in params or not isinstance(params.get("budget"), (int, float)) or params.get("budget") < 0:
                return False, "budget variant requires non-negative 'budget' parameter"
        if variant == "max_distance":
            if "max_assignment_distance" not in params or params.get("max_assignment_distance") is None or params.get("max_assignment_distance") <= 0:
                return False, "max_distance variant requires positive 'max_assignment_distance'"
        
        return True, None
    
    def solve(
        self,
        data: Dict[str, gpd.GeoDataFrame],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        distance_metric: str = "network"
    ) -> Dict[str, Any]:
        """Solve the P-Median problem"""
        start_time = time.time()
        
        try:
            # Import optimizer
            from utils.distance_calculator import DistanceCalculator
            from utils.heuristics.genetic_solver import PMedianGeneticSolver, GAConfig
            
            # Extract data
            demand_gdf = data.get('demand_points')
            candidate_gdf = data.get('candidate_sites')
            
            if demand_gdf is None or candidate_gdf is None:
                raise ValueError("Both demand_points and candidate_sites are required")
            
            # Get parameters
            p = parameters['n_facilities']
            objective_type = parameters.get('objective', 'total')
            variant = parameters.get('variant', 'base')
            capacities = parameters.get('capacities') if isinstance(parameters.get('capacities'), (list, np.ndarray)) else None
            facility_costs = parameters.get('facility_costs') if isinstance(parameters.get('facility_costs'), (list, np.ndarray)) else None
            budget = parameters.get('budget') if isinstance(parameters.get('budget'), (int, float)) else None
            max_assign_dist = parameters.get('max_assignment_distance') if isinstance(parameters.get('max_assignment_distance'), (int, float)) else None
            service_radius_unit = parameters.get('service_radius_unit', 'm')
            
            # Fallback configuration
            fallback_time_limit = float(parameters.get('fallback_time_limit_seconds', 60.0))
            use_ga_after_timeout = True
            logger.info(f"P-Median: Fallback time limit set to {fallback_time_limit:.2f} seconds")
            
            # Merge fixed_open/fixed_closed/existing_facilities into constraints
            constraints = self._merge_facility_set_constraints(constraints, parameters)

            # Validate p
            if p > len(candidate_gdf):
                raise ValueError(f"Cannot locate {p} facilities with only {len(candidate_gdf)} candidate sites")
            
            # Get demand weights (population or provided demand column)
            demand_weights = self._extract_weights(demand_gdf, parameters)
            
            # Calculate distance matrix
            dist_calc = DistanceCalculator()
            network_graph = data.get('_network_graph')
            distance_matrix = dist_calc.calculate_distance_matrix(
                demand_gdf, candidate_gdf, metric=distance_metric, network_graph=network_graph
            )
            # For max-distance variant, compute a mask where assignments allowed
            distance_mask = None
            if variant == 'max_distance' and max_assign_dist is not None:
                try:
                    distance_mask = (distance_matrix <= float(max_assign_dist)).astype(int)
                except Exception:
                    distance_mask = None
            
            # Solve using optimization (respect a time limit for fallback orchestration)
            mip_start = time.time()
            solution = self._solve_mip(
                distance_matrix=distance_matrix,
                demand_weights=demand_weights,
                p=p,
                constraints=constraints,
                variant=variant,
                objective_type=objective_type,
                capacities=np.array(capacities, dtype=float) if capacities is not None else None,
                facility_costs=np.array(facility_costs, dtype=float) if facility_costs is not None else None,
                budget=float(budget) if budget is not None else None,
                distance_mask=distance_mask,
                time_limit_seconds=fallback_time_limit
            )
            mip_elapsed = time.time() - mip_start

            # If timed out or we reached the 60s window, switch to GA per user choice (a)
            timed_out_flag = bool(solution.get('solver_details', {}).get('timed_out', False))
            # The GA fallback only optimises the *base* P-Median objective; it does
            # not honour the capacitated / budget / max_distance constraints, so it
            # must not be substituted for those variants (it would silently return a
            # constraint-violating solution labelled "feasible"). Keep the MIP
            # incumbent for unsupported variants instead.
            ga_supported = (variant == 'base')
            ga_needed = use_ga_after_timeout and (timed_out_flag or mip_elapsed >= max(0.1, 0.95 * fallback_time_limit))
            logger.info(f"P-Median timeout check: mip_elapsed={mip_elapsed:.2f}s, fallback_limit={fallback_time_limit:.2f}s, timed_out_flag={timed_out_flag}, ga_needed={ga_needed}, ga_supported={ga_supported}")
            if ga_needed and ga_supported:
                logger.info("P-Median: Falling back to Genetic Algorithm")
                logger.info(f"P-Median: MIP solver status: {solution.get('status', 'unknown')}, objective: {solution.get('objective_value', 'N/A')}")
                incumbent_mask = None
                if solution.get('selected_facilities'):
                    incumbent_mask = np.zeros(distance_matrix.shape[1], dtype=int)
                    for j in solution['selected_facilities']:
                        if 0 <= int(j) < incumbent_mask.size:
                            incumbent_mask[int(j)] = 1
                ga_cfg = GAConfig(time_limit_seconds=float(parameters.get('ga_time_budget_seconds', 60.0)))
                logger.info(f"P-Median: Starting GA with time budget: {ga_cfg.time_limit_seconds:.2f} seconds")
                ga = PMedianGeneticSolver(ga_cfg)
                ga_result = ga.solve(
                    distance_matrix=distance_matrix,
                    demand_weights=demand_weights,
                    p=p,
                    objective_type=objective_type,
                    initial_solution=incumbent_mask,
                    time_budget_seconds=ga_cfg.time_limit_seconds
                )
                logger.info(f"P-Median: GA completed with status: {ga_result.get('status', 'unknown')}, objective: {ga_result.get('objective_value', 'N/A')}")
                ga_details = {
                    **ga_result.get("solver_details", {}),
                    "fallback_from": solution.get('solver_details', {}).get('solver', 'mip'),
                    "fallback_reason": "time_limit"
                }
                solution = {
                    "status": ga_result.get("status", "feasible"),
                    "objective_value": float(ga_result["objective_value"]),
                    "selected_facilities": ga_result["selected_facilities"],
                    "assignments": ga_result["assignments"],
                    "solver_details": ga_details
                }
            elif ga_needed and not ga_supported:
                logger.info(f"P-Median: GA fallback not available for variant '{variant}'; keeping MIP incumbent")
                if timed_out_flag:
                    solution.setdefault('warnings', []).append(
                        f"MIP time limit reached for the '{variant}' P-Median variant; returning best "
                        f"incumbent (genetic-algorithm fallback is not available for this variant)."
                    )
            else:
                logger.info(f"P-Median: MIP solver completed successfully within time limit, no fallback needed. Status: {solution.get('status', 'unknown')}, objective: {solution.get('objective_value', 'N/A')}")
            
            # Validate assignments against service radius if provided
            validation_results = self._validate_assignments(
                distance_matrix, solution['assignments'], parameters
            )
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                distance_matrix, demand_weights, 
                solution['selected_facilities'], 
                solution['assignments'],
                objective_type,
                distance_unit=service_radius_unit
            )
            
            # Add validation results to metrics
            if validation_results:
                metrics.update(validation_results)

            # Variant-specific metrics
            try:
                if variant == 'budget' and facility_costs is not None:
                    sel = solution.get('selected_facilities', [])
                    metrics['budget_used'] = float(sum(float(facility_costs[j]) for j in sel if 0 <= j < len(facility_costs)))
                if variant == 'capacitated' and capacities is not None:
                    sel = solution.get('selected_facilities', [])
                    served_weight_by_fac = {j: 0.0 for j in sel}
                    for i, j in solution.get('assignments', {}).items():
                        if j in served_weight_by_fac:
                            served_weight_by_fac[j] += float(demand_weights[i])
                    util = {}
                    for j in sel:
                        cap = float(capacities[j]) if 0 <= j < len(capacities) else 0.0
                        util[j] = (served_weight_by_fac.get(j, 0.0) / cap) if cap > 0 else None
                    metrics['capacity_utilization'] = util
            except Exception:
                pass
            
            solution_time = time.time() - start_time
            
            return {
                "status": solution['status'],
                "objective_value": solution['objective_value'],
                "selected_facilities": solution['selected_facilities'],
                "assignments": solution['assignments'],
                "metrics": metrics,
                "solution_time": solution_time,
                "solver_details": solution.get('solver_details', {}),
                "warnings": solution.get('warnings', []),
                "academic_metadata": {
                    "algorithm_used": ("Genetic Algorithm (GA)"
                                       if solution.get('solver_details', {}).get('solver') == 'ga'
                                       else "Mixed Integer Programming (MIP)"),
                    "references": self.get_metadata()['academic_refs'][:2],
                    "assumptions": [
                        "Each demand point is assigned to exactly one facility",
                        "Facilities have unlimited capacity" if parameters.get('variant', 'base') != 'capacitated' else "Facilities have capacity limits",
                        f"Distance metric: {distance_metric}",
                        "Travel occurs along straight lines" if distance_metric == "euclidean" else f"Distance calculation: {distance_metric}"
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error solving P-Median problem: {e}")
            return {
                "status": "error",
                "error": str(e),
                "solution_time": time.time() - start_time
            }
    
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
    
    def _solve_mip(
        self,
        distance_matrix: np.ndarray,
        demand_weights: np.ndarray,
        p: int,
        constraints: Dict[str, Any],
        variant: str,
        objective_type: str,
        capacities: Optional[np.ndarray],
        facility_costs: Optional[np.ndarray],
        budget: Optional[float],
        distance_mask: Optional[np.ndarray],
        time_limit_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Solve using Mixed Integer Programming"""
        n_demand, n_candidates = distance_matrix.shape
        
        # Try Gurobi first, fall back to PuLP
        try:
            import gurobipy as gp
            from gurobipy import GRB
            return self._solve_gurobi(
                distance_matrix, demand_weights, p, constraints,
                variant, objective_type, capacities, facility_costs, budget, distance_mask,
                time_limit_seconds=time_limit_seconds
            )
        except ImportError:
            logger.info("Gurobi not available, using PuLP")
            return self._solve_pulp(
                distance_matrix, demand_weights, p, constraints,
                variant, objective_type, capacities, facility_costs, budget, distance_mask,
                time_limit_seconds=time_limit_seconds
            )
    
    def _solve_gurobi(
        self,
        distance_matrix: np.ndarray,
        demand_weights: np.ndarray,
        p: int,
        constraints: Dict[str, Any],
        variant: str,
        objective_type: str,
        capacities: Optional[np.ndarray],
        facility_costs: Optional[np.ndarray],
        budget: Optional[float],
        distance_mask: Optional[np.ndarray],
        time_limit_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Solve using Gurobi"""
        import gurobipy as gp
        from gurobipy import GRB

        from .base_solver import configure_gurobi_model

        n_demand, n_candidates = distance_matrix.shape

        model = gp.Model("p-median")
        configure_gurobi_model(model, time_limit_seconds)
        if time_limit_seconds is not None:
            logger.info(
                "P-Median Gurobi: TimeLimit set to %.2f seconds", float(time_limit_seconds)
            )

        x = model.addVars(n_candidates, vtype=GRB.BINARY, name="x")  # facility location
        y = model.addVars(n_demand, n_candidates, vtype=GRB.BINARY, name="y")  # assignment
        
        # Objective: minimize weighted distance (total or average)
        if objective_type == "average":
            # For average distance, we minimize total weighted distance divided by total weight
            # Since total weight is constant, this is equivalent to minimizing total weighted distance
            total_weight = sum(demand_weights)
            obj = gp.quicksum(
                distance_matrix[i, j] * demand_weights[i] * y[i, j]
                for i in range(n_demand)
                for j in range(n_candidates)
            ) / total_weight if total_weight > 0 else 0
        else:  # objective_type == "total"
            # For total distance, minimize total weighted distance
            obj = gp.quicksum(
                distance_matrix[i, j] * demand_weights[i] * y[i, j]
                for i in range(n_demand)
                for j in range(n_candidates)
            )
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraint: locate exactly p facilities
        model.addConstr(gp.quicksum(x[j] for j in range(n_candidates)) == p, "p_facilities")

        # Constraint: each demand assigned to exactly one facility (batched).
        model.addConstrs(
            (gp.quicksum(y[i, j] for j in range(n_candidates)) == 1
             for i in range(n_demand)),
            name="assign_demand",
        )

        # Constraint: assignment only to open facilities (batched → ~10× faster
        # to build than n_demand·n_candidates individual addConstr calls).
        model.addConstrs(
            (y[i, j] <= x[j] for i in range(n_demand) for j in range(n_candidates)),
            name="open_facility",
        )

        # Max-distance: forbid assignments beyond threshold
        if variant == 'max_distance' and distance_mask is not None:
            model.addConstrs(
                (y[i, j] == 0 for i in range(n_demand) for j in range(n_candidates)
                 if distance_mask[i, j] == 0),
                name="maxdist_forbid",
            )

        # Capacitated variant
        if variant == 'capacitated' and capacities is not None:
            if len(capacities) != n_candidates:
                raise ValueError("capacities length must match number of candidate sites")
            model.addConstrs(
                (gp.quicksum(demand_weights[i] * y[i, j] for i in range(n_demand))
                 <= float(capacities[j]) * x[j]
                 for j in range(n_candidates)),
                name="capacity",
            )

        # Budget variant
        if variant == 'budget':
            if budget is None:
                raise ValueError("budget variant requires 'budget'")
            if facility_costs is None or len(facility_costs) != n_candidates:
                raise ValueError("facility_costs length must match number of candidate sites for budget variant")
            model.addConstr(
                gp.quicksum(float(facility_costs[j]) * x[j] for j in range(n_candidates)) <= float(budget),
                "budget_limit"
            )

        # Add custom constraints
        must_include = constraints.get('must_include', [])
        for j in must_include:
            if 0 <= j < n_candidates:
                model.addConstr(x[j] == 1, f"must_include_{j}")
        
        must_exclude = constraints.get('must_exclude', [])
        for j in must_exclude:
            if 0 <= j < n_candidates:
                model.addConstr(x[j] == 0, f"must_exclude_{j}")
        
        # Solve
        model.optimize()
        
        # Extract solution
        timed_out = (model.status == GRB.TIME_LIMIT)
        if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL or model.status == GRB.TIME_LIMIT:
            selected = [j for j in range(n_candidates) if x[j].X > 0.5]
            assignments = {}
            for i in range(n_demand):
                for j in range(n_candidates):
                    if y[i, j].X > 0.5:
                        assignments[i] = j
                        break
            
            # Calculate the actual total weighted distance from assignments
            total_weighted_distance = 0.0
            for i, j in assignments.items():
                total_weighted_distance += distance_matrix[i, j] * demand_weights[i]
            
            # Return the objective value that corresponds to what was actually optimized
            if objective_type == "average":
                total_weight = sum(demand_weights)
                objective_value = total_weighted_distance / total_weight if total_weight > 0 else 0
            else:  # objective_type == "total"
                objective_value = total_weighted_distance
            
            return {
                'status': 'optimal' if model.status == GRB.OPTIMAL else 'feasible',
                'objective_value': objective_value,  # Return the actual objective that was optimized
                'selected_facilities': selected,
                'assignments': assignments,
                'solver_details': {
                    'solver': 'gurobi',
                    'gap': model.MIPGap,
                    'iterations': model.IterCount,
                    'formulation': f'P-Median MIP ({variant})',
                    'total_weighted_distance': total_weighted_distance,  # Always include total for reference
                    'solver_objective_value': model.objVal,  # Keep the solver's objective value for reference
                    'timed_out': bool(timed_out)
                }
            }
        else:
            return {
                'status': 'infeasible',
                'objective_value': None,
                'selected_facilities': [],
                'assignments': {},
                'solver_details': {'solver': 'gurobi', 'status': model.status}
            }
    
    def _solve_pulp(
        self,
        distance_matrix: np.ndarray,
        demand_weights: np.ndarray,
        p: int,
        constraints: Dict[str, Any],
        variant: str,
        objective_type: str,
        capacities: Optional[np.ndarray],
        facility_costs: Optional[np.ndarray],
        budget: Optional[float],
        distance_mask: Optional[np.ndarray],
        time_limit_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Solve using PuLP"""
        import pulp
        
        n_demand, n_candidates = distance_matrix.shape
        
        # Create problem
        prob = pulp.LpProblem("p-median", pulp.LpMinimize)
        
        # Decision variables
        x = pulp.LpVariable.dicts("x", range(n_candidates), cat='Binary')
        y = pulp.LpVariable.dicts("y", 
            ((i, j) for i in range(n_demand) for j in range(n_candidates)),
            cat='Binary')
        
        # Objective: minimize weighted distance (total or average)
        if objective_type == "average":
            # For average distance, we minimize total weighted distance divided by total weight
            # Since total weight is constant, this is equivalent to minimizing total weighted distance
            total_weight = sum(demand_weights)
            if total_weight > 0:
                prob += pulp.lpSum([
                    distance_matrix[i, j] * demand_weights[i] * y[(i, j)]
                    for i in range(n_demand)
                    for j in range(n_candidates)
                ]) / total_weight
            else:
                prob += pulp.lpSum([
                    distance_matrix[i, j] * demand_weights[i] * y[(i, j)]
                    for i in range(n_demand)
                    for j in range(n_candidates)
                ])
        else:  # objective_type == "total"
            # For total distance, minimize total weighted distance
            prob += pulp.lpSum([
                distance_matrix[i, j] * demand_weights[i] * y[(i, j)]
                for i in range(n_demand)
                for j in range(n_candidates)
            ])
        
        # Constraints
        prob += pulp.lpSum([x[j] for j in range(n_candidates)]) == p, "p_facilities"
        
        for i in range(n_demand):
            prob += pulp.lpSum([y[(i, j)] for j in range(n_candidates)]) == 1, f"assign_{i}"
        
        for i in range(n_demand):
            for j in range(n_candidates):
                prob += y[(i, j)] <= x[j], f"open_{i}_{j}"
                if variant == 'max_distance' and distance_mask is not None and distance_mask[i, j] == 0:
                    prob += y[(i, j)] == 0
        
        if variant == 'capacitated' and capacities is not None:
            if len(capacities) != n_candidates:
                raise ValueError("capacities length must match number of candidate sites")
            for j in range(n_candidates):
                prob += pulp.lpSum([demand_weights[i] * y[(i, j)] for i in range(n_demand)]) <= float(capacities[j]) * x[j]
        
        if variant == 'budget':
            if budget is None:
                raise ValueError("budget variant requires 'budget'")
            if facility_costs is None or len(facility_costs) != n_candidates:
                raise ValueError("facility_costs length must match number of candidate sites for budget variant")
            prob += pulp.lpSum([float(facility_costs[j]) * x[j] for j in range(n_candidates)]) <= float(budget), "budget_limit"
        
        # Custom constraints
        must_include = constraints.get('must_include', [])
        for j in must_include:
            if 0 <= j < n_candidates:
                prob += x[j] == 1
        
        must_exclude = constraints.get('must_exclude', [])
        for j in must_exclude:
            if 0 <= j < n_candidates:
                prob += x[j] == 0
        
        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=float(time_limit_seconds) if time_limit_seconds is not None else None)
        prob.solve(solver)
        
        # Extract solution
        if prob.status == pulp.LpStatusOptimal:
            selected = [j for j in range(n_candidates) if pulp.value(x[j]) > 0.5]
            assignments = {}
            for i in range(n_demand):
                for j in range(n_candidates):
                    if pulp.value(y[(i, j)]) > 0.5:
                        assignments[i] = j
                        break
            
            # Calculate the actual total weighted distance from assignments
            total_weighted_distance = 0.0
            for i, j in assignments.items():
                total_weighted_distance += distance_matrix[i, j] * demand_weights[i]
            
            # Return the objective value that corresponds to what was actually optimized
            if objective_type == "average":
                total_weight = sum(demand_weights)
                objective_value = total_weighted_distance / total_weight if total_weight > 0 else 0
            else:  # objective_type == "total"
                objective_value = total_weighted_distance
            
            return {
                'status': 'optimal',
                'objective_value': objective_value,  # Return the actual objective that was optimized
                'selected_facilities': selected,
                'assignments': assignments,
                'solver_details': {
                    'solver': 'pulp',
                    'formulation': 'P-Median MIP',
                    'total_weighted_distance': total_weighted_distance,  # Always include total for reference
                    'solver_objective_value': pulp.value(prob.objective),  # Keep the solver's objective value for reference
                    'timed_out': False
                }
            }
        else:
            # Not optimal: approximate timeout if a time limit was set
            timed_out = bool(time_limit_seconds is not None)
            return {
                'status': 'infeasible',
                'objective_value': None,
                'selected_facilities': [],
                'assignments': {},
                'solver_details': {'solver': 'pulp', 'status': prob.status, 'timed_out': timed_out}
            }
    
    def _validate_assignments(
        self,
        distance_matrix: np.ndarray,
        assignments: Dict[int, int],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate assignments against service radius constraints.
        Returns validation results including violations and warnings.
        """
        validation_results = {}
        
        # Check for service radius parameter
        service_radius = parameters.get('service_radius')
        max_assignment_distance = parameters.get('max_assignment_distance')
        
        if service_radius is None and max_assignment_distance is None:
            return validation_results
        
        # Use the appropriate threshold value
        threshold_value = service_radius if service_radius is not None else max_assignment_distance
        
        # Get unit from parameters and convert to meters for comparison
        # (distance_matrix is already in meters from DistanceCalculator)
        from utils.distance_calculator import DistanceCalculator
        dist_calc = DistanceCalculator()
        
        service_radius_unit = parameters.get('service_radius_unit')
        threshold_meters = dist_calc._convert_to_meters(threshold_value, service_radius_unit)
        
        violations = []
        violation_distances = []
        
        for demand_idx, facility_idx in assignments.items():
            if demand_idx < distance_matrix.shape[0] and facility_idx < distance_matrix.shape[1]:
                distance = distance_matrix[demand_idx, facility_idx]
                if distance > threshold_meters:
                    violations.append({
                        'demand_idx': demand_idx,
                        'facility_idx': facility_idx,
                        'distance': distance,
                        'threshold': threshold_meters,
                        'excess': distance - threshold_meters
                    })
                    violation_distances.append(distance)
        
        if violations:
            validation_results['assignment_violations'] = violations
            validation_results['violation_count'] = len(violations)
            validation_results['max_violation_distance'] = max(violation_distances) if violation_distances else 0
            validation_results['avg_violation_distance'] = sum(violation_distances) / len(violation_distances) if violation_distances else 0
            
            # Log warnings for violations
            unit_str = service_radius_unit or 'meters'
            logger.warning(f"Found {len(violations)} assignment violations exceeding service radius {threshold_value} {unit_str} ({threshold_meters:.0f} meters)")
            for violation in violations[:5]:  # Log first 5 violations
                logger.warning(f"Demand {violation['demand_idx']} -> Facility {violation['facility_idx']}: "
                              f"distance {violation['distance']:.2f}m > threshold {violation['threshold']:.2f}m")
        
        return validation_results

    def _calculate_metrics(
        self,
        distance_matrix: np.ndarray,
        demand_weights: np.ndarray,
        selected_facilities: List[int],
        assignments: Dict[int, int],
        objective_type: str,
        distance_unit: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate solution metrics"""
        # Calculate distances for assignments
        distances = []
        for demand_id, facility_id in assignments.items():
            dist = distance_matrix[demand_id, facility_id]
            weight = demand_weights[demand_id]
            distances.append((dist, weight))
        
        total_weighted_distance = float(sum(d * w for d, w in distances))
        total_weight = float(sum(demand_weights))
        average_distance = total_weighted_distance / total_weight if total_weight > 0 else 0.0
        max_distance = float(max((d for d, w in distances), default=0))
        
        # Convert distances to user-requested units if specified
        from utils.distance_calculator import DistanceCalculator
        dist_calc = DistanceCalculator()
        
        # Determine objective value based on objective type
        if objective_type == "average":
            objective_value = average_distance
            objective_name = "average_weighted_distance"
        else:  # "total" is default
            objective_value = total_weighted_distance
            objective_name = "total_weighted_distance"
            
        # Perform unit conversions
        objective_value = dist_calc.convert_meters_to_unit(objective_value, distance_unit)
        total_weighted_distance = dist_calc.convert_meters_to_unit(total_weighted_distance, distance_unit)
        average_distance = dist_calc.convert_meters_to_unit(average_distance, distance_unit)
        max_distance = dist_calc.convert_meters_to_unit(max_distance, distance_unit)
        
        return {
            # Objective info for P-Median
            "objective_value": objective_value,
            "objective_name": objective_name,
            "objective_type": objective_type,
            # Core metrics
            "total_weighted_distance": total_weighted_distance,
            "average_distance": average_distance,
            "max_distance": max_distance,
            "num_facilities": len(selected_facilities),
            "num_demand_points": len(assignments),
            "total_demand_weight": total_weight
        }
    
    def explain_solution(
        self,
        solution: Dict[str, Any],
        data: Dict[str, gpd.GeoDataFrame],
        detail_level: str = "standard",
        objective_type: str = "total"
    ) -> str:
        """Generate human-readable explanation"""
        if solution.get('status') == 'error':
            return f"❌ Solution failed: {solution.get('error', 'Unknown error')}"
        
        if solution.get('status') == 'infeasible':
            return "❌ No feasible solution found. This may be due to conflicting constraints."
        
        metrics = solution.get('metrics', {})
        n_facilities = metrics.get('num_facilities', 0)
        avg_dist = metrics.get('average_distance', 0)
        max_dist = metrics.get('max_distance', 0)
        total_dist = metrics.get('total_weighted_distance', 0)
        
        if detail_level == "brief":
            return f"Located {n_facilities} facilities with average distance {avg_dist:.2f}."
        
        elif detail_level == "standard":
            objective_desc = "total weighted distance" if objective_type == "total" else "average weighted distance"
            return f"""
**P-Median Solution Summary**

✅ Successfully located {n_facilities} facilities to minimize {objective_desc}.

**Key Metrics:**
- Total Weighted Distance: {total_dist:.2f}
- Average Distance: {avg_dist:.2f}
- Maximum Distance: {max_dist:.2f}

The selected facilities minimize the overall access cost, with each demand point assigned to its most appropriate facility.
            """.strip()
        
        elif detail_level == "detailed":
            selected = solution.get('selected_facilities', [])
            return f"""
**P-Median Solution - Detailed Analysis**

✅ **Optimization Status:** {solution.get('status', 'Unknown')}
✅ **Solution Time:** {solution.get('solution_time', 0):.2f} seconds

**Selected Facilities:** {len(selected)}
Facility indices: {', '.join(map(str, selected))}

**Distance Metrics:**
- Total Weighted Distance: {total_dist:.2f}
- Average Distance per Demand Unit: {avg_dist:.2f}
- Maximum Distance (worst case): {max_dist:.2f}

**Problem Characteristics:**
- Demand Points Served: {metrics.get('num_demand_points', 0)}
- Total Demand Weight: {metrics.get('total_demand_weight', 0):.2f}

**Interpretation:**
This solution minimizes the total weighted distance between demand points and facilities. The selected locations balance proximity to high-demand areas while maintaining service coverage across all demand points.
            """.strip()
        
        elif detail_level == "academic":
            solver_details = solution.get('solver_details', {})
            academic = solution.get('academic_metadata', {})
            return f"""
**P-Median Problem Solution - Academic Report**

**Mathematical Formulation:**
The P-Median problem minimizes Σᵢ Σⱼ dᵢⱼ · wᵢ · yᵢⱼ subject to locating exactly p facilities.

**Solution Details:**
- Status: {solution.get('status', 'Unknown')}
- Objective Value: {solution.get('objective_value', 0):.4f}
- Solution Time: {solution.get('solution_time', 0):.4f} seconds
- Solver: {solver_details.get('solver', 'Unknown')}
- Gap: {solver_details.get('gap', 0):.4f}

**Results:**
- Facilities Located: {n_facilities}
- Average Distance: {avg_dist:.4f}
- Maximum Distance: {max_dist:.4f}
- Total Demand Served: {metrics.get('total_demand_weight', 0):.2f}

**Methodology:**
{academic.get('algorithm_used', 'MIP')} formulation solved using {'Gurobi' if solver_details.get('solver') == 'gurobi' else 'PuLP CBC'}.

**Assumptions:**
{chr(10).join('- ' + a for a in academic.get('assumptions', []))}

**Key References:**
{chr(10).join('- ' + r for r in academic.get('references', []))}
            """.strip()
        
        return "Solution explanation not available."

