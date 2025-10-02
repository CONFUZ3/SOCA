from .base_solver import SpatialOptimizationProblem
from typing import Dict, List, Any, Optional
import geopandas as gpd
import numpy as np
import time
import logging

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
            "variants": [
                "Vertex P-Center",
                "Absolute P-Center",
                "Weighted P-Center",
                "Conditional P-Center"
            ]
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
            
            if p > len(candidate_gdf):
                raise ValueError(f"Cannot locate {p} facilities with only {len(candidate_gdf)} candidate sites")
            
            # Calculate distance matrix
            dist_calc = DistanceCalculator()
            distance_matrix = dist_calc.calculate_distance_matrix(
                demand_gdf, candidate_gdf, metric=distance_metric
            )
            
            # Solve using MIP
            solution = self._solve_mip(distance_matrix, p, constraints)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                distance_matrix, 
                solution['selected_facilities'],
                solution['assignments']
            )
            
            solution_time = time.time() - start_time
            
            return {
                "status": solution['status'],
                "objective_value": solution['objective_value'],
                "selected_facilities": solution['selected_facilities'],
                "assignments": solution['assignments'],
                "metrics": metrics,
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
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve using MIP"""
        try:
            import gurobipy as gp
            from gurobipy import GRB
            return self._solve_gurobi(distance_matrix, p, constraints)
        except ImportError:
            logger.info("Gurobi not available, using PuLP")
            return self._solve_pulp(distance_matrix, p, constraints)
    
    def _solve_gurobi(
        self,
        distance_matrix: np.ndarray,
        p: int,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        import gurobipy as gp
        from gurobipy import GRB
        
        n_demand, n_candidates = distance_matrix.shape
        
        model = gp.Model("p-center")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 300)
        
        # Decision variables
        x = model.addVars(n_candidates, vtype=GRB.BINARY, name="x")
        y = model.addVars(n_demand, n_candidates, vtype=GRB.BINARY, name="y")
        W = model.addVar(vtype=GRB.CONTINUOUS, name="W")  # Maximum distance
        
        # Objective: minimize maximum distance
        model.setObjective(W, GRB.MINIMIZE)
        
        # Constraints
        model.addConstr(gp.quicksum(x[j] for j in range(n_candidates)) == p)
        
        for i in range(n_demand):
            model.addConstr(gp.quicksum(y[i, j] for j in range(n_candidates)) == 1)
        
        for i in range(n_demand):
            for j in range(n_candidates):
                model.addConstr(y[i, j] <= x[j])
                # W must be at least as large as any assigned distance
                model.addConstr(W >= distance_matrix[i, j] * y[i, j])
        
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
                    'formulation': 'P-Center Minimax MIP'
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
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        import pulp
        
        n_demand, n_candidates = distance_matrix.shape
        
        prob = pulp.LpProblem("p-center", pulp.LpMinimize)
        
        x = pulp.LpVariable.dicts("x", range(n_candidates), cat='Binary')
        y = pulp.LpVariable.dicts("y",
            ((i, j) for i in range(n_demand) for j in range(n_candidates)),
            cat='Binary')
        W = pulp.LpVariable("W", lowBound=0)
        
        prob += W
        
        prob += pulp.lpSum([x[j] for j in range(n_candidates)]) == p
        
        for i in range(n_demand):
            prob += pulp.lpSum([y[(i, j)] for j in range(n_candidates)]) == 1
        
        for i in range(n_demand):
            for j in range(n_candidates):
                prob += y[(i, j)] <= x[j]
                prob += W >= distance_matrix[i, j] * y[(i, j)]
        
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
                for j in range(n_candidates):
                    if pulp.value(y[(i, j)]) > 0.5:
                        assignments[i] = j
                        break
            
            return {
                'status': 'optimal',
                'objective_value': pulp.value(W),
                'selected_facilities': selected,
                'assignments': assignments,
                'solver_details': {'solver': 'pulp', 'formulation': 'P-Center Minimax MIP'}
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
        distance_matrix: np.ndarray,
        selected_facilities: List[int],
        assignments: Dict[int, int]
    ) -> Dict[str, float]:
        distances = [distance_matrix[i, j] for i, j in assignments.items()]
        
        return {
            "max_distance": max(distances) if distances else 0,
            "average_distance": np.mean(distances) if distances else 0,
            "min_distance": min(distances) if distances else 0,
            "std_distance": np.std(distances) if distances else 0,
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

