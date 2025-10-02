from .base_solver import SpatialOptimizationProblem
from typing import Dict, List, Any, Optional
import geopandas as gpd
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class MCLPSolver(SpatialOptimizationProblem):
    """
    Maximum Covering Location Problem (MCLP) Solver
    
    Maximizes demand coverage within a distance threshold using p facilities.
    """
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Maximum Covering Location Problem (MCLP)",
            "short_name": "mclp",
            "category": "coverage maximization",
            "description": "Locate p facilities to maximize the demand covered within a specified service distance threshold. Optimal when you have a fixed budget (number of facilities) and want to maximize population served.",
            "mathematical_formulation": """
Maximize: Σᵢ wᵢ · zᵢ

Subject to:
- Σⱼ xⱼ = p  (locate exactly p facilities)
- zᵢ ≤ Σⱼ∈Nᵢ xⱼ, ∀i  (coverage only if nearby facility exists)
- xⱼ, zᵢ ∈ {0,1}

Where:
- wᵢ = weight/demand at point i
- zᵢ = 1 if demand i is covered, 0 otherwise
- xⱼ = 1 if facility located at j
- Nᵢ = set of candidates within threshold distance of demand i
- p = number of facilities
            """,
            "academic_refs": [
                "Church, R., & ReVelle, C. (1974). The maximal covering location problem. Papers of the Regional Science Association, 32(1), 101-118.",
                "Daskin, M. S. (2013). Network and discrete location: models, algorithms, and applications. John Wiley & Sons.",
                "Schilling, D. A., Jayaraman, V., & Barkhi, R. (1993). A review of covering problems in facility location. Location Science, 1(1), 25-55."
            ],
            "complexity": "NP-hard",
            "typical_use_cases": [
                "Emergency service coverage (police, ambulance)",
                "Retail location with service radius",
                "WiFi access point placement",
                "Sensor network design",
                "Healthcare facility placement with access standards"
            ],
            "keywords": [
                "mclp", "maximum cover", "maximize coverage", "service radius",
                "threshold", "within distance", "coverage", "service area"
            ],
            "variants": [
                "MCLP with mandatory closeness",
                "Backup coverage MCLP",
                "Gradual coverage MCLP",
                "MCLP with facility capacity"
            ]
        }
    
    def get_conversation_prompts(self) -> Dict[str, Any]:
        return {
            "problem_detection": [
                "maximize coverage", "maximum cover", "service radius",
                "threshold", "within", "mclp", "covered", "coverage"
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
                    "param": "service_radius",
                    "question": "What is the maximum service distance/radius?",
                    "type": "float",
                    "validation": "Must be a positive number",
                    "help": "Demand is considered covered if within this distance from a facility"
                }
            ],
            "constraint_suggestions": [
                "Would you like to specify any facilities that must be included?",
                "Should certain demand points have higher priority?"
            ],
            "explanation_template": "The MCLP solution locates {n_facilities} facilities to maximize coverage within {service_radius}. Coverage: {coverage_pct:.1f}%."
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
            return False, "Missing required parameter: n_facilities"
        
        if "service_radius" not in params:
            return False, "Missing required parameter: service_radius"
        
        if not isinstance(params["n_facilities"], int) or params["n_facilities"] <= 0:
            return False, "n_facilities must be a positive integer"
        
        if not isinstance(params["service_radius"], (int, float)) or params["service_radius"] <= 0:
            return False, "service_radius must be a positive number"
        
        return True, None
    
    def solve(
        self,
        data: Dict[str, gpd.GeoDataFrame],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        distance_metric: str = "euclidean"
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            from utils.distance_calculator import DistanceCalculator
            
            demand_gdf = data.get('demand_points')
            candidate_gdf = data.get('candidate_sites')
            
            if demand_gdf is None or candidate_gdf is None:
                raise ValueError("Both demand_points and candidate_sites are required")
            
            p = parameters['n_facilities']
            service_radius = parameters['service_radius']
            
            if p > len(candidate_gdf):
                raise ValueError(f"Cannot locate {p} facilities with only {len(candidate_gdf)} candidate sites")
            
            # Get demand weights
            demand_weights = self._extract_weights(demand_gdf)
            
            # Calculate coverage matrix
            dist_calc = DistanceCalculator()
            coverage_matrix = dist_calc.calculate_coverage_matrix(
                demand_gdf, candidate_gdf, 
                threshold=service_radius,
                metric=distance_metric
            )
            
            # Solve
            solution = self._solve_mip(coverage_matrix, demand_weights, p, constraints)
            
            # Calculate distance matrix for metrics
            distance_matrix = dist_calc.calculate_distance_matrix(
                demand_gdf, candidate_gdf, metric=distance_metric
            )
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                coverage_matrix, distance_matrix, demand_weights,
                solution['selected_facilities'],
                service_radius
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
                "academic_metadata": {
                    "algorithm_used": "Mixed Integer Programming (MIP)",
                    "references": self.get_metadata()['academic_refs'][:2],
                    "assumptions": [
                        f"Service radius: {service_radius}",
                        "Demand covered if within service radius of any facility",
                        f"Distance metric: {distance_metric}",
                        "Facilities have unlimited capacity"
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error solving MCLP: {e}")
            return {
                "status": "error",
                "error": str(e),
                "solution_time": time.time() - start_time
            }
    
    def _extract_weights(self, demand_gdf: gpd.GeoDataFrame) -> np.ndarray:
        weight_cols = ['demand', 'weight', 'population', 'pop']
        for col in weight_cols:
            if col in demand_gdf.columns:
                weights = demand_gdf[col].values
                if np.all(weights > 0):
                    return weights
        return np.ones(len(demand_gdf))
    
    def _solve_mip(
        self,
        coverage_matrix: np.ndarray,
        demand_weights: np.ndarray,
        p: int,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            return self._solve_gurobi(coverage_matrix, demand_weights, p, constraints)
        except ImportError:
            return self._solve_pulp(coverage_matrix, demand_weights, p, constraints)
    
    def _solve_gurobi(
        self,
        coverage_matrix: np.ndarray,
        demand_weights: np.ndarray,
        p: int,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        import gurobipy as gp
        from gurobipy import GRB
        
        n_demand, n_candidates = coverage_matrix.shape
        
        model = gp.Model("mclp")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 300)
        
        # Decision variables
        x = model.addVars(n_candidates, vtype=GRB.BINARY, name="x")  # facility location
        z = model.addVars(n_demand, vtype=GRB.BINARY, name="z")  # demand covered
        
        # Objective: maximize weighted coverage
        obj = gp.quicksum(demand_weights[i] * z[i] for i in range(n_demand))
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Constraint: locate exactly p facilities
        model.addConstr(gp.quicksum(x[j] for j in range(n_candidates)) == p)
        
        # Constraint: coverage only if nearby facility exists
        for i in range(n_demand):
            covering_facilities = [j for j in range(n_candidates) if coverage_matrix[i, j] == 1]
            if covering_facilities:
                model.addConstr(
                    z[i] <= gp.quicksum(x[j] for j in covering_facilities),
                    f"cover_{i}"
                )
            else:
                model.addConstr(z[i] == 0)  # Can't be covered
        
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
        
        if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
            selected = [j for j in range(n_candidates) if x[j].X > 0.5]
            
            # Determine assignments (each demand to nearest selected facility)
            assignments = {}
            for i in range(n_demand):
                if z[i].X > 0.5:
                    # Find nearest selected facility that covers this demand
                    for j in selected:
                        if coverage_matrix[i, j] == 1:
                            assignments[i] = j
                            break
            
            return {
                'status': 'optimal' if model.status == GRB.OPTIMAL else 'feasible',
                'objective_value': model.objVal,
                'selected_facilities': selected,
                'assignments': assignments,
                'solver_details': {
                    'solver': 'gurobi',
                    'gap': model.MIPGap,
                    'formulation': 'MCLP MIP'
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
        demand_weights: np.ndarray,
        p: int,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        import pulp
        
        n_demand, n_candidates = coverage_matrix.shape
        
        prob = pulp.LpProblem("mclp", pulp.LpMaximize)
        
        x = pulp.LpVariable.dicts("x", range(n_candidates), cat='Binary')
        z = pulp.LpVariable.dicts("z", range(n_demand), cat='Binary')
        
        # Objective
        prob += pulp.lpSum([demand_weights[i] * z[i] for i in range(n_demand)])
        
        # Constraints
        prob += pulp.lpSum([x[j] for j in range(n_candidates)]) == p
        
        for i in range(n_demand):
            covering_facilities = [j for j in range(n_candidates) if coverage_matrix[i, j] == 1]
            if covering_facilities:
                prob += z[i] <= pulp.lpSum([x[j] for j in covering_facilities])
            else:
                prob += z[i] == 0
        
        must_include = constraints.get('must_include', [])
        for j in must_include:
            if 0 <= j < n_candidates:
                prob += x[j] == 1
        
        must_exclude = constraints.get('must_exclude', [])
        for j in must_exclude:
            if 0 <= j < n_candidates:
                prob += x[j] == 0
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            selected = [j for j in range(n_candidates) if pulp.value(x[j]) > 0.5]
            
            assignments = {}
            for i in range(n_demand):
                if pulp.value(z[i]) > 0.5:
                    for j in selected:
                        if coverage_matrix[i, j] == 1:
                            assignments[i] = j
                            break
            
            return {
                'status': 'optimal',
                'objective_value': pulp.value(prob.objective),
                'selected_facilities': selected,
                'assignments': assignments,
                'solver_details': {'solver': 'pulp', 'formulation': 'MCLP MIP'}
            }
        else:
            return {
                'status': 'infeasible',
                'objective_value': None,
                'selected_facilities': [],
                'assignments': {}
            }
    
    def _calculate_metrics(
        self,
        coverage_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        demand_weights: np.ndarray,
        selected_facilities: List[int],
        service_radius: float
    ) -> Dict[str, float]:
        n_demand = len(demand_weights)
        
        # Determine which demands are covered
        covered = np.zeros(n_demand, dtype=bool)
        for j in selected_facilities:
            covered |= (coverage_matrix[:, j] == 1)
        
        covered_weight = np.sum(demand_weights[covered])
        total_weight = np.sum(demand_weights)
        coverage_pct = (covered_weight / total_weight * 100) if total_weight > 0 else 0
        
        # Calculate average distance for covered demands
        covered_indices = np.where(covered)[0]
        if len(covered_indices) > 0:
            min_distances = np.min(distance_matrix[covered_indices][:, selected_facilities], axis=1)
            avg_distance_covered = np.mean(min_distances)
        else:
            avg_distance_covered = 0
        
        return {
            "coverage_percentage": coverage_pct,
            "covered_demand": covered_weight,
            "total_demand": total_weight,
            "uncovered_demand": total_weight - covered_weight,
            "num_covered_points": int(np.sum(covered)),
            "num_uncovered_points": int(n_demand - np.sum(covered)),
            "service_radius": service_radius,
            "average_distance_covered": avg_distance_covered,
            "num_facilities": len(selected_facilities)
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
        coverage_pct = metrics.get('coverage_percentage', 0)
        service_radius = metrics.get('service_radius', 0)
        
        if detail_level == "brief":
            return f"Located {n_facilities} facilities covering {coverage_pct:.1f}% of demand."
        
        else:
            return f"""
**MCLP Solution Summary**

✅ Successfully located {n_facilities} facilities to maximize coverage within service radius of {service_radius:.2f}.

**Coverage Metrics:**
- Coverage: {coverage_pct:.1f}% of total demand
- Covered Demand: {metrics.get('covered_demand', 0):.1f}
- Uncovered Demand: {metrics.get('uncovered_demand', 0):.1f}
- Covered Points: {metrics.get('num_covered_points', 0)} / {metrics.get('num_covered_points', 0) + metrics.get('num_uncovered_points', 0)}

**Performance:**
- Average Distance (covered demands): {metrics.get('average_distance_covered', 0):.2f}

The solution maximizes the population served within the specified service radius.
            """.strip()
    
    def get_visualization_config(self) -> Dict[str, Any]:
        config = super().get_visualization_config()
        config['show_service_areas'] = True
        return config

