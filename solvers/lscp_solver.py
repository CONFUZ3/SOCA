from .base_solver import SpatialOptimizationProblem
from typing import Dict, List, Any, Optional
import geopandas as gpd
import numpy as np
import time
import logging

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
            "variants": [
                "LSCP with backup coverage",
                "Conditional set covering",
                "Maximal covering with optional coverage",
                "Probabilistic LSCP"
            ]
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
            
            service_radius = parameters['service_radius']
            
            # Calculate coverage matrix
            dist_calc = DistanceCalculator()
            coverage_matrix = dist_calc.calculate_coverage_matrix(
                demand_gdf, candidate_gdf,
                threshold=service_radius,
                metric=distance_metric
            )
            
            # Check feasibility
            uncoverable = []
            for i in range(len(demand_gdf)):
                if not np.any(coverage_matrix[i, :]):
                    uncoverable.append(i)
            
            if uncoverable:
                return {
                    "status": "infeasible",
                    "error": f"Infeasible: {len(uncoverable)} demand points cannot be covered with service radius {service_radius}. Consider increasing the service radius or adding more candidate sites.",
                    "uncoverable_points": uncoverable,
                    "solution_time": time.time() - start_time
                }
            
            # Solve
            solution = self._solve_mip(coverage_matrix, constraints)
            
            # Calculate distance matrix for metrics
            distance_matrix = dist_calc.calculate_distance_matrix(
                demand_gdf, candidate_gdf, metric=distance_metric
            )
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                coverage_matrix, distance_matrix,
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
                    "algorithm_used": "Mixed Integer Programming (Set Cover)",
                    "references": self.get_metadata()['academic_refs'][:2],
                    "assumptions": [
                        f"Service radius: {service_radius}",
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
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            return self._solve_gurobi(coverage_matrix, constraints)
        except ImportError:
            return self._solve_pulp(coverage_matrix, constraints)
    
    def _solve_gurobi(
        self,
        coverage_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        import gurobipy as gp
        from gurobipy import GRB
        
        n_demand, n_candidates = coverage_matrix.shape
        
        model = gp.Model("lscp")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 300)
        
        # Decision variables
        x = model.addVars(n_candidates, vtype=GRB.BINARY, name="x")
        
        # Objective: minimize number of facilities
        model.setObjective(gp.quicksum(x[j] for j in range(n_candidates)), GRB.MINIMIZE)
        
        # Constraint: each demand must be covered by at least one facility
        for i in range(n_demand):
            covering_facilities = [j for j in range(n_candidates) if coverage_matrix[i, j] == 1]
            if covering_facilities:
                model.addConstr(
                    gp.quicksum(x[j] for j in covering_facilities) >= 1,
                    f"cover_{i}"
                )
            else:
                # This should have been caught in feasibility check
                raise ValueError(f"Demand point {i} cannot be covered")
        
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
        
        if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
            selected = [j for j in range(n_candidates) if x[j].X > 0.5]
            
            # Determine assignments
            assignments = {}
            for i in range(n_demand):
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
                    'formulation': 'LSCP Set Cover MIP'
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
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        import pulp
        
        n_demand, n_candidates = coverage_matrix.shape
        
        prob = pulp.LpProblem("lscp", pulp.LpMinimize)
        
        x = pulp.LpVariable.dicts("x", range(n_candidates), cat='Binary')
        
        # Objective
        prob += pulp.lpSum([x[j] for j in range(n_candidates)])
        
        # Constraints
        for i in range(n_demand):
            covering_facilities = [j for j in range(n_candidates) if coverage_matrix[i, j] == 1]
            if covering_facilities:
                prob += pulp.lpSum([x[j] for j in covering_facilities]) >= 1
            else:
                raise ValueError(f"Demand point {i} cannot be covered")
        
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
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            selected = [j for j in range(n_candidates) if pulp.value(x[j]) > 0.5]
            
            assignments = {}
            for i in range(n_demand):
                covering_selected = [j for j in selected if coverage_matrix[i, j] == 1]
                if covering_selected:
                    assignments[i] = covering_selected[0]
            
            return {
                'status': 'optimal',
                'objective_value': len(selected),
                'selected_facilities': selected,
                'assignments': assignments,
                'solver_details': {'solver': 'pulp', 'formulation': 'LSCP Set Cover MIP'}
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
        selected_facilities: List[int],
        service_radius: float
    ) -> Dict[str, float]:
        n_demand = coverage_matrix.shape[0]
        
        # All demands should be covered
        covered = np.zeros(n_demand, dtype=bool)
        for j in selected_facilities:
            covered |= (coverage_matrix[:, j] == 1)
        
        coverage_pct = (np.sum(covered) / n_demand * 100) if n_demand > 0 else 0
        
        # Calculate distances
        if len(selected_facilities) > 0:
            min_distances = np.min(distance_matrix[:, selected_facilities], axis=1)
            avg_distance = np.mean(min_distances)
            max_distance = np.max(min_distances)
        else:
            avg_distance = 0
            max_distance = 0
        
        return {
            "num_facilities": len(selected_facilities),
            "coverage_percentage": coverage_pct,
            "num_covered_points": int(np.sum(covered)),
            "num_uncovered_points": int(n_demand - np.sum(covered)),
            "service_radius": service_radius,
            "average_distance": avg_distance,
            "max_distance": max_distance,
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

