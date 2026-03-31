from .base_solver import SpatialOptimizationProblem
from typing import Dict, List, Any, Optional
import geopandas as gpd
import numpy as np
import time
import logging

from utils.heuristics.genetic_solver import GAConfig, MCLPGeneticSolver

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
                "classical",
                "budget",
                "capacitated",
                "probabilistic",
                "multi_coverage",
                "backup",
                "hierarchical",
                "dynamic"
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
                },
                {
                    "param": "variant",
                    "question": "Which MCLP variant? (classical, budget, capacitated, probabilistic, multi_coverage, backup)",
                    "type": "str",
                    "validation": "One of the supported variants",
                    "help": "Defaults to classical if not specified"
                },
                {
                    "param": "capacities",
                    "question": "What are the capacity limits for each facility? (required for capacitated variant)",
                    "type": "list",
                    "validation": "List of positive numbers, one per candidate site",
                    "help": "Required for capacitated variant. Represents max demand each facility can serve. Can be provided as parameter, in candidate data columns (capacity, cap, max_service, throughput), or calculated from demand dataset population"
                },
                {
                    "param": "budget",
                    "question": "What is the total budget for facility establishment? (required for budget variant)",
                    "type": "float",
                    "validation": "Must be a positive number",
                    "help": "Required for budget variant. Total cost constraint for facility selection"
                },
                {
                    "param": "facility_costs",
                    "question": "What are the establishment costs for each facility? (optional for budget variant)",
                    "type": "list",
                    "validation": "List of non-negative numbers, one per candidate site",
                    "help": "Optional for budget variant. Can be provided as parameter or in candidate data columns (cost, facility_cost, open_cost)"
                },
                {
                    "param": "k_coverage",
                    "question": "How many facilities should cover each demand point? (for multi_coverage/backup variants)",
                    "type": "int",
                    "validation": "Must be a positive integer",
                    "help": "Required for multi_coverage and backup variants. Minimum number of facilities covering each demand point"
                },
                {
                    "param": "facility_reliability",
                    "question": "What are the reliability probabilities for each facility? (for probabilistic variant)",
                    "type": "list",
                    "validation": "List of numbers between 0 and 1, one per candidate site",
                    "help": "Optional for probabilistic variant. Reliability probability for each facility (defaults to 1.0)"
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
        variant = params.get("variant", "classical")
        
        if "service_radius" not in params:
            return False, "Missing required parameter: service_radius"
        if not isinstance(params["service_radius"], (int, float)) or params["service_radius"] <= 0:
            return False, "service_radius must be a positive number"
        
        if variant == "classical" or variant == "multi_coverage" or variant == "backup" or variant == "probabilistic":
            if "n_facilities" not in params:
                return False, "Missing required parameter: n_facilities"
            if not isinstance(params["n_facilities"], int) or params["n_facilities"] <= 0:
                return False, "n_facilities must be a positive integer"
        elif variant == "budget":
            if "budget" not in params:
                return False, "Missing required parameter for budget variant: budget"
            if not isinstance(params["budget"], (int, float)) or params["budget"] <= 0:
                return False, "budget must be a positive number"
        elif variant == "capacitated":
            # Capacitated can work with either fixed p or budget; require at least one of them
            if ("n_facilities" not in params) and ("budget" not in params):
                return False, "Capacitated variant requires n_facilities or budget"
        else:
            return False, f"Unsupported variant: {variant}"
        
        # Multi-coverage parameters
        if variant in ("multi_coverage", "backup"):
            k = params.get("k_coverage", 2 if variant == "backup" else 1)
            if not isinstance(k, int) or k <= 0:
                return False, "k_coverage must be a positive integer"
        
        return True, None
    
    def solve(
        self,
        data: Dict[str, gpd.GeoDataFrame],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        distance_metric: str = "euclidean"
    ) -> Dict[str, Any]:
        """
        Main solve method that delegates to variant-specific solvers.
        
        Uses strategy pattern to select appropriate variant solver.
        """
        start_time = time.time()
        
        try:
            # Validate inputs and prepare shared data
            shared_data = self._prepare_shared_data(data, parameters, distance_metric)
            fallback_time_limit = float(parameters.get('fallback_time_limit_seconds', 60.0))
            ga_time_budget = float(parameters.get('ga_time_budget_seconds', 60.0))
            logger.info(f"MCLP: Fallback time limit set to {fallback_time_limit:.2f} seconds, GA time budget: {ga_time_budget:.2f} seconds")
            
            # Get service radius unit for visualization consistency
            service_radius_unit = shared_data.get('service_radius_unit', 'm')
            
            variant = parameters.get('variant', 'classical')
            logger.info(f"MCLP Solver: Using variant '{variant}' with parameters: {parameters}")
            
            # Select variant-specific solver
            mip_start = time.time()
            solution = self._solve_variant(
                variant=variant,
                shared_data=shared_data,
                parameters=parameters,
                constraints=constraints,
                time_limit_seconds=fallback_time_limit
            )
            mip_elapsed = time.time() - mip_start
            
            timed_out_flag = bool(solution.get('solver_details', {}).get('timed_out', False))
            ga_needed = timed_out_flag or (
                fallback_time_limit > 0 and mip_elapsed >= max(0.1, 0.95 * fallback_time_limit)
            )
            logger.info(f"MCLP timeout check: mip_elapsed={mip_elapsed:.2f}s, fallback_limit={fallback_time_limit:.2f}s, timed_out_flag={timed_out_flag}, ga_needed={ga_needed}")
            if ga_needed:
                logger.info(f"MCLP: Falling back to Genetic Algorithm for variant '{variant}'")
                logger.info(f"MCLP: MIP solver status: {solution.get('status', 'unknown')}, objective: {solution.get('objective_value', 'N/A')}")
                ga_solver = MCLPGeneticSolver(GAConfig(time_limit_seconds=ga_time_budget))
                logger.info(f"MCLP: Starting GA with time budget: {ga_time_budget:.2f} seconds")
                if ga_solver.supports_variant(variant):
                    incumbent_mask = None
                    selected = solution.get('selected_facilities')
                    if selected:
                        incumbent_mask = np.zeros(shared_data['coverage_matrix'].shape[1], dtype=np.int8)
                        for idx in selected:
                            if 0 <= idx < incumbent_mask.size:
                                incumbent_mask[idx] = 1
                    try:
                        ga_result = ga_solver.solve(
                            coverage_matrix=shared_data['coverage_matrix'],
                            distance_matrix=shared_data['distance_matrix'],
                            demand_weights=shared_data['demand_weights'],
                            variant=variant,
                            p=parameters.get('n_facilities'),
                            facility_costs=shared_data.get('facility_costs'),
                            budget=parameters.get('budget'),
                            k_coverage=parameters.get('k_coverage', 2 if variant == 'backup' else 1),
                            reliability=parameters.get('facility_reliability'),
                            initial_solution=incumbent_mask,
                            time_budget_seconds=ga_time_budget
                        )
                        logger.info(f"MCLP: GA completed with status: {ga_result.get('status', 'unknown')}, objective: {ga_result.get('objective_value', 'N/A')}")
                        ga_details = {
                            **ga_result.get("solver_details", {}),
                            "fallback_from": solution.get('solver_details', {}).get('solver', 'mip'),
                            "fallback_reason": "time_limit"
                        }
                        solution = {
                            "status": ga_result.get("status", "feasible"),
                            "objective_value": ga_result["objective_value"],
                            "selected_facilities": ga_result["selected_facilities"],
                            "assignments": ga_result.get("assignments", {}),
                            "z_values": ga_result.get("z_values"),
                            "solver_details": ga_details
                        }
                    except Exception as ga_err:
                        logger.error(f"MCLP: GA fallback for variant '{variant}' failed: {ga_err}", exc_info=True)
                    else:
                        logger.warning(
                            "GA fallback not available for MCLP variant '%s'; returning MIP result",
                            variant
                        )
                else:
                    logger.info(f"MCLP: MIP solver completed successfully within time limit, no fallback needed. Status: {solution.get('status', 'unknown')}, objective: {solution.get('objective_value', 'N/A')}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                shared_data['coverage_matrix'], 
                shared_data['distance_matrix'], 
                shared_data['demand_weights'],
                solution['selected_facilities'],
                shared_data['service_radius'],
                variant,
                parameters.get('k_coverage', 2 if variant == 'backup' else 1),
                solution.get('assignments', {}),
                solution.get('z_values', {}),
                shared_data.get('capacities'),
                shared_data.get('facility_costs'),
                parameters.get('budget'),
                parameters.get('facility_reliability'),
                solution.get('y_values'),
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
                "service_radius_unit": service_radius_unit,  # Pass unit for visualization
                "academic_metadata": {
                    "algorithm_used": "Mixed Integer Programming (MIP)",
                    "references": self.get_metadata()['academic_refs'][:2],
                    "assumptions": [
                        f"Service radius: {shared_data['service_radius']} {service_radius_unit}",
                        "Demand covered if within service radius of any facility",
                        f"Distance metric: {distance_metric}",
                        f"MCLP variant: {variant}",
                        "Facilities have unlimited capacity" if variant != "capacitated" else "Facilities have capacity constraints"
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
    
    # ============================================================================
    # SHARED INFRASTRUCTURE METHODS
    # ============================================================================
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
    
    def _prepare_shared_data(
        self,
        data: Dict[str, gpd.GeoDataFrame],
        parameters: Dict[str, Any],
        distance_metric: str
    ) -> Dict[str, Any]:
        """
        Prepare shared data structures used by all variants.
        
        Returns:
            Dictionary containing coverage_matrix, distance_matrix, demand_weights,
            and other shared data structures.
        """
        from utils.distance_calculator import DistanceCalculator
        
        demand_gdf = data.get('demand_points')
        candidate_gdf = data.get('candidate_sites')
        
        if demand_gdf is None or candidate_gdf is None:
            raise ValueError("Both demand_points and candidate_sites are required")
        
        service_radius = float(parameters['service_radius'])
        # Get explicit unit from parameters (default to 'm' if not specified)
        service_radius_unit = parameters.get('service_radius_unit', 'm')
        
        # Get demand weights
        demand_weights = self._extract_weights(demand_gdf, parameters)
        
        # Calculate coverage and distance matrices
        dist_calc = DistanceCalculator()
        
        # Get unit conversion info
        unit_info = dist_calc.get_unit_info(service_radius, service_radius_unit)
        
        coverage_matrix = dist_calc.calculate_coverage_matrix(
            demand_gdf, candidate_gdf, 
            threshold=service_radius,
            metric=distance_metric,
            unit=service_radius_unit
        )
        
        distance_matrix = dist_calc.calculate_distance_matrix(
            demand_gdf, candidate_gdf, metric=distance_metric
        )
        
        # Extract optional parameters
        facility_costs = self._extract_facility_costs(candidate_gdf, parameters)
        capacities = self._extract_facility_capacities(candidate_gdf, parameters)
        
        return {
            'demand_gdf': demand_gdf,
            'candidate_gdf': candidate_gdf,
            'coverage_matrix': coverage_matrix,
            'distance_matrix': distance_matrix,
            'demand_weights': demand_weights,
            'service_radius': service_radius,
            'service_radius_unit': service_radius_unit,
            'facility_costs': facility_costs,
            'capacities': capacities,
            'unit_info': unit_info
        }
    
    def _solve_variant(
        self,
        variant: str,
        shared_data: Dict[str, Any],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Delegate to variant-specific solver using strategy pattern.
        """
        solver_map = {
            'classical': self._solve_classical,
            'budget': self._solve_budget,
            'capacitated': self._solve_capacitated,
            'probabilistic': self._solve_probabilistic,
            'multi_coverage': self._solve_multi_coverage,
            'backup': self._solve_backup
        }
        
        if variant not in solver_map:
            raise ValueError(f"Unsupported MCLP variant: {variant}")
        
        return solver_map[variant](shared_data, parameters, constraints, time_limit_seconds)
    
    def _extract_facility_costs(
        self,
        candidate_gdf: gpd.GeoDataFrame,
        parameters: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract facility costs from parameters or candidate data."""
        # Priority: explicit parameter, then candidate columns, else None
        if 'facility_costs' in parameters and parameters['facility_costs'] is not None:
            arr = np.asarray(parameters['facility_costs'], dtype=float)
            if len(arr) != len(candidate_gdf):
                raise ValueError("Length of facility_costs must match number of candidate sites")
            if np.any(arr < 0):
                raise ValueError("facility_costs must be non-negative")
            return arr
        for col in ['cost', 'facility_cost', 'open_cost']:
            if col in candidate_gdf.columns:
                values = candidate_gdf[col].astype(float).to_numpy()
                if np.any(values < 0):
                    raise ValueError(f"Candidate column '{col}' contains negative costs")
                return values
        return None

    def _extract_facility_capacities(
        self,
        candidate_gdf: gpd.GeoDataFrame,
        parameters: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract facility capacities from parameters or candidate data."""
        if 'capacities' in parameters and parameters['capacities'] is not None:
            arr = np.asarray(parameters['capacities'], dtype=float)
            if len(arr) != len(candidate_gdf):
                raise ValueError("Length of capacities must match number of candidate sites")
            if np.any(arr < 0):
                raise ValueError("capacities must be non-negative")
            return arr
        for col in ['capacity', 'cap', 'max_service']:
            if col in candidate_gdf.columns:
                values = candidate_gdf[col].astype(float).to_numpy()
                if np.any(values < 0):
                    raise ValueError(f"Candidate column '{col}' contains negative capacities")
                return values
        return None

    def _extract_weights(self, demand_gdf: gpd.GeoDataFrame, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Extract demand weights from GeoDataFrame.
        
        Returns:
            Array of demand weights, allowing zero weights for valid use cases.
        """
        # DEBUG: Log available columns
        all_cols = [c for c in demand_gdf.columns if c != 'geometry']
        logger.info(f"_extract_weights: Available columns in demand data: {all_cols}")
        
        # 1) Explicit parameter takes precedence
        try:
            explicit_col = parameters.get('demand_weight_column')
            if explicit_col:
                logger.info(f"_extract_weights: Looking for explicit column '{explicit_col}'")
                explicit_lower = str(explicit_col).lower()
                # Try exact case-insensitive match first
                for c in demand_gdf.columns:
                    if c.lower() == explicit_lower:
                        values = demand_gdf[c].astype(float).to_numpy()
                        if np.any(values < 0):
                            raise ValueError(f"Demand weight column '{c}' contains negative values")
                        logger.info(f"Using explicit weight column '{c}' with sum={values.sum():.2f}")
                        return values
                # Try partial match (for truncated column names like 'ExpectedVa' from 'ExpectedValue')
                for c in demand_gdf.columns:
                    c_lower = c.lower()
                    if c_lower.startswith(explicit_lower[:6]) or explicit_lower.startswith(c_lower[:6]):
                        try:
                            values = demand_gdf[c].astype(float).to_numpy()
                            if np.all(values >= 0):
                                logger.info(f"Using partial-matched weight column '{c}' for requested '{explicit_col}' with sum={values.sum():.2f}")
                                return values
                        except Exception:
                            continue
                logger.warning(f"Explicit demand_weight_column '{explicit_col}' not found in columns: {all_cols}")
        except Exception as e:
            logger.warning(f"Failed to use explicit demand_weight_column: {e}")

        # 2) Case-insensitive exact matches of common names
        common_exact = ['demand', 'weight', 'population', 'pop']
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

        # 3) Substring heuristic: pick the first numeric column whose name contains pop/weight/demand/expected/value
        substr_keys = ['population', 'pop', 'weight', 'demand', 'expected', 'value', 'score', 'priority']
        for c in demand_gdf.columns:
            lc = c.lower()
            if any(k in lc for k in substr_keys):
                try:
                    values = demand_gdf[c].astype(float).to_numpy()
                    if np.all(values >= 0):
                        return values
                except Exception:
                    continue

        # 4) Fallback to ones if nothing found
        logger.warning("No suitable demand weight column found; defaulting to 1.0 per demand point")
        return np.ones(len(demand_gdf))
    
    def _calculate_assignments(
        self,
        selected_facilities: List[int],
        coverage_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        variant: str
    ) -> Dict[int, int]:
        """
        Calculate assignments of demand points to selected facilities.
        
        Assigns each demand point to the nearest selected facility that can cover it.
        """
        assignments = {}
        n_demand = coverage_matrix.shape[0]
        
        # For all variants, assign all covered demand to nearest facility
        for i in range(n_demand):
            covering_selected = [j for j in selected_facilities if coverage_matrix[i, j] == 1]
            if covering_selected:
                distances = [distance_matrix[i, j] for j in covering_selected]
                nearest_idx = np.argmin(distances)
                assignments[i] = covering_selected[nearest_idx]
        
        return assignments
    
    # ============================================================================
    # VARIANT-SPECIFIC SOLVERS
    # ============================================================================
    
    def _solve_classical(
        self,
        shared_data: Dict[str, Any],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float]
    ) -> Dict[str, Any]:
        """Solve classical MCLP variant."""
        p = int(parameters['n_facilities'])
        if p > len(shared_data['candidate_gdf']):
            raise ValueError(f"Cannot locate {p} facilities with only {len(shared_data['candidate_gdf'])} candidate sites")
        
        return self._solve_mip(
            coverage_matrix=shared_data['coverage_matrix'],
            demand_weights=shared_data['demand_weights'],
            p=p,
            constraints=constraints,
            variant='classical',
            facility_costs=shared_data['facility_costs'],
            budget=None,
            capacities=None,
            k_coverage=1,
            reliability=None,
            distance_matrix=shared_data['distance_matrix'],
            time_limit_seconds=time_limit_seconds
        )
    
    def _solve_budget(
        self,
        shared_data: Dict[str, Any],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float]
    ) -> Dict[str, Any]:
        """Solve budget-constrained MCLP variant."""
        budget = float(parameters['budget'])
        
        return self._solve_mip(
            coverage_matrix=shared_data['coverage_matrix'],
            demand_weights=shared_data['demand_weights'],
            p=None,
            constraints=constraints,
            variant='budget',
            facility_costs=shared_data['facility_costs'],
            budget=budget,
            capacities=None,
            k_coverage=1,
            reliability=None,
            distance_matrix=shared_data['distance_matrix'],
            time_limit_seconds=time_limit_seconds
        )
    
    def _solve_capacitated(
        self,
        shared_data: Dict[str, Any],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float]
    ) -> Dict[str, Any]:
        """Solve capacitated MCLP variant."""
        capacities = shared_data['capacities']
        
        # Handle missing capacities with default calculation
        if capacities is None:
            total_demand = float(np.sum(shared_data['demand_weights']))
            p = parameters.get('n_facilities', 5)  # Default to 5 if not specified
            default_capacity = total_demand / max(1, p)
            capacities = np.full(len(shared_data['candidate_gdf']), default_capacity)
            logger.warning(f"Capacitated variant: Using default capacity of {default_capacity:.2f} per facility")
        
        # Log capacity information
        logger.info(f"Capacitated MCLP - Capacity values: {capacities}")
        logger.info(f"Total demand: {float(np.sum(shared_data['demand_weights'])):.2f}")
        logger.info(f"Total capacity: {float(np.sum(capacities)):.2f}")
        
        p = parameters.get('n_facilities')
        if p is not None:
            p = int(p)
        
        return self._solve_mip(
            coverage_matrix=shared_data['coverage_matrix'],
            demand_weights=shared_data['demand_weights'],
            p=p,
            constraints=constraints,
            variant='capacitated',
            facility_costs=shared_data['facility_costs'],
            budget=parameters.get('budget'),
            capacities=capacities,
            k_coverage=1,
            reliability=None,
            distance_matrix=shared_data['distance_matrix'],
            time_limit_seconds=time_limit_seconds
        )
    
    def _solve_probabilistic(
        self,
        shared_data: Dict[str, Any],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float]
    ) -> Dict[str, Any]:
        """Solve probabilistic MCLP variant."""
        p = int(parameters['n_facilities'])
        reliability = parameters.get('facility_reliability')
        
        return self._solve_mip(
            coverage_matrix=shared_data['coverage_matrix'],
            demand_weights=shared_data['demand_weights'],
            p=p,
            constraints=constraints,
            variant='probabilistic',
            facility_costs=shared_data['facility_costs'],
            budget=None,
            capacities=None,
            k_coverage=1,
            reliability=reliability,
            distance_matrix=shared_data['distance_matrix'],
            time_limit_seconds=time_limit_seconds
        )
    
    def _solve_multi_coverage(
        self,
        shared_data: Dict[str, Any],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float]
    ) -> Dict[str, Any]:
        """Solve multi-coverage MCLP variant."""
        p = int(parameters['n_facilities'])
        k_coverage = int(parameters.get('k_coverage', 2))
        
        return self._solve_mip(
            coverage_matrix=shared_data['coverage_matrix'],
            demand_weights=shared_data['demand_weights'],
            p=p,
            constraints=constraints,
            variant='multi_coverage',
            facility_costs=shared_data['facility_costs'],
            budget=None,
            capacities=None,
            k_coverage=k_coverage,
            reliability=None,
            distance_matrix=shared_data['distance_matrix'],
            time_limit_seconds=time_limit_seconds
        )
    
    def _solve_backup(
        self,
        shared_data: Dict[str, Any],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        time_limit_seconds: Optional[float]
    ) -> Dict[str, Any]:
        """Solve backup coverage MCLP variant."""
        p = int(parameters['n_facilities'])
        k_coverage = int(parameters.get('k_coverage', 2))
        
        return self._solve_mip(
            coverage_matrix=shared_data['coverage_matrix'],
            demand_weights=shared_data['demand_weights'],
            p=p,
            constraints=constraints,
            variant='backup',
            facility_costs=shared_data['facility_costs'],
            budget=None,
            capacities=None,
            k_coverage=k_coverage,
            reliability=None,
            distance_matrix=shared_data['distance_matrix'],
            time_limit_seconds=time_limit_seconds
        )
    
    # ============================================================================
    # MIP SOLVER INFRASTRUCTURE
    # ============================================================================
    
    def _solve_mip(
        self,
        coverage_matrix: np.ndarray,
        demand_weights: np.ndarray,
        p: Optional[int],
        constraints: Dict[str, Any],
        variant: str,
        facility_costs: Optional[np.ndarray],
        budget: Optional[float],
        capacities: Optional[np.ndarray],
        k_coverage: int,
        reliability: Optional[np.ndarray],
        distance_matrix: np.ndarray,
        time_limit_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Solve MCLP using Mixed Integer Programming.
        
        Attempts to use Gurobi first (commercial-grade solver with better performance),
        falls back to PuLP (open-source solver) if Gurobi is not available.
        """
        try:
            import gurobipy as gp
            from gurobipy import GRB
            return self._solve_gurobi(
                coverage_matrix, demand_weights, p, constraints,
                variant, facility_costs, budget, capacities, k_coverage, reliability, distance_matrix,
                time_limit_seconds
            )
        except ImportError:
            return self._solve_pulp(
                coverage_matrix, demand_weights, p, constraints,
                variant, facility_costs, budget, capacities, k_coverage, reliability, distance_matrix,
                time_limit_seconds
            )
    
    def _solve_gurobi(
        self,
        coverage_matrix: np.ndarray,
        demand_weights: np.ndarray,
        p: Optional[int],
        constraints: Dict[str, Any],
        variant: str,
        facility_costs: Optional[np.ndarray],
        budget: Optional[float],
        capacities: Optional[np.ndarray],
        k_coverage: int,
        reliability: Optional[np.ndarray],
        distance_matrix: np.ndarray,
        time_limit_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        import gurobipy as gp
        from gurobipy import GRB
        
        n_demand, n_candidates = coverage_matrix.shape
        
        model = gp.Model("mclp")
        model.setParam('OutputFlag', 0)
        if time_limit_seconds is not None:
            model.setParam('TimeLimit', float(time_limit_seconds))
            logger.info(f"MCLP Gurobi: Setting TimeLimit to {time_limit_seconds:.2f} seconds")
        else:
            model.setParam('TimeLimit', 300)
        model.setParam('MIPGap', 0.01)  # 1% optimality gap
        model.setParam('Presolve', 2)   # Aggressive presolve
        model.setParam('Cuts', 2)       # Aggressive cut generation
        
        # Decision variables
        x = model.addVars(n_candidates, vtype=GRB.BINARY, name="x")  # facility location
        z_vtype = GRB.CONTINUOUS if variant == "probabilistic" else GRB.BINARY
        z = model.addVars(n_demand, vtype=z_vtype, lb=0.0, ub=1.0, name="z")  # demand covered or fraction covered
        
        # For capacitated variant, add assignment variables
        if variant == "capacitated":
            # Fractional assignment variables: fraction of demand i served by facility j (0..1)
            y = model.addVars(n_demand, n_candidates, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y")
        
        # Objective: maximize weighted coverage
        # Capacitated: maximize total served population directly via y
        if variant == "capacitated":
            obj = gp.quicksum(float(demand_weights[i]) * y[i, j] for i in range(n_demand) for j in range(n_candidates) if coverage_matrix[i, j] == 1)
        else:
            # Non-capacitated and probabilistic variants use z
            obj = gp.quicksum(float(demand_weights[i]) * z[i] for i in range(n_demand))
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Facility selection constraints
        if variant == "budget":
            if facility_costs is None:
                facility_costs = np.ones(n_candidates)
            model.addConstr(gp.quicksum(facility_costs[j] * x[j] for j in range(n_candidates)) <= float(budget))
        else:
            if p is None:
                raise ValueError("n_facilities is required for this variant")
            model.addConstr(gp.quicksum(x[j] for j in range(n_candidates)) == int(p))
        
        # Coverage constraints per variant
        logger.info(f"MCLP Gurobi: Applying {variant} variant constraints")
        for i in range(n_demand):
            covering_facilities = [j for j in range(n_candidates) if coverage_matrix[i, j] == 1]
            if not covering_facilities:
                model.addConstr(z[i] == 0)
                continue
            if variant == "probabilistic":
                # Linear upper bound using reliability r_j: z_i <= sum r_j x_j
                rel = self._normalize_reliability(reliability, n_candidates)
                model.addConstr(z[i] <= gp.quicksum(float(rel[j]) * x[j] for j in covering_facilities), f"prob_cover_{i}")
            elif variant == "multi_coverage" or variant == "backup":
                # Require at least k facilities to claim coverage
                model.addConstr(gp.quicksum(x[j] for j in covering_facilities) >= int(k_coverage) * z[i], f"kcover_{i}")
            elif variant == "capacitated":
                # For capacitated variant, coverage indicator z[i] is 1 if any positive fraction served
                model.addConstr(z[i] <= gp.quicksum(y[i, j] for j in covering_facilities), f"cover_{i}")
                # Assignment constraints: can only serve from open facilities and total fraction ≤ 1
                for j in covering_facilities:
                    model.addConstr(y[i, j] <= x[j], f"assign_{i}_{j}")
                model.addConstr(gp.quicksum(y[i, j] for j in covering_facilities) <= 1.0, f"fraction_sum_{i}")
            else:
                # classical and budget: binary coverage if any facility covers
                model.addConstr(z[i] <= gp.quicksum(x[j] for j in covering_facilities), f"cover_{i}")
        
        # Capacitated facility capacity constraints
        # Constraint: Total demand weight served by facility j ≤ Capacity of facility j
        if variant == "capacitated":
            if capacities is None:
                raise ValueError("Capacitated variant requires facility capacities")
            for j in range(n_candidates):
                served_demand_j = gp.quicksum(
                    float(demand_weights[i]) * y[i, j] for i in range(n_demand)
                    if coverage_matrix[i, j] == 1
                )
                model.addConstr(served_demand_j <= float(capacities[j]) * x[j], f"cap_{j}")

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
            
            # Determine assignments and served fractions for capacitated
            if variant == "capacitated":
                assignments = {}
                y_values = {}
                for i in range(n_demand):
                    frac_sum = 0.0
                    best_j = None
                    best_frac = 0.0
                    for j in range(n_candidates):
                        if coverage_matrix[i, j] == 1:
                            frac = float(y[i, j].X)
                            if frac > 0.0:
                                y_values[(i, j)] = frac
                                frac_sum += frac
                                if frac > best_frac:
                                    best_frac = frac
                                    best_j = j
                    if best_j is not None:
                        assignments[i] = best_j
                
                # Log capacity utilization for capacitated variant
                if capacities is not None:
                    logger.info(f"Capacitated MCLP - Selected facilities: {selected}")
                    for j in selected:
                        served_demand = sum(
                            float(demand_weights[i]) * float(y[i, j].X)
                            for i in range(n_demand) if coverage_matrix[i, j] == 1
                        )
                        capacity_j = float(capacities[j])
                        utilization = (served_demand / capacity_j * 100) if capacity_j > 0 else 0
                        logger.info(f"Facility {j}: Served Demand={served_demand:.2f}, Capacity={capacity_j:.2f}, Utilization={utilization:.1f}%")
            else:
                # For non-capacitated variants, use the standard assignment method
                assignments = self._calculate_assignments(
                    selected, coverage_matrix, distance_matrix, variant
                )
            
            # Extract z values and y values for metrics calculation
            z_values = {i: z[i].X for i in range(n_demand)}
            
            result = {
                'status': 'optimal' if model.status == GRB.OPTIMAL else 'feasible',
                'objective_value': model.objVal,
                'selected_facilities': selected,
                'assignments': assignments,
                'z_values': z_values,
                'solver_details': {
                    'solver': 'gurobi',
                    'gap': model.MIPGap,
                    'formulation': f'MCLP {variant} MIP',
                    'timed_out': bool(timed_out)
                }
            }
            if variant == 'capacitated':
                result['y_values'] = y_values
            return result
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
        p: Optional[int],
        constraints: Dict[str, Any],
        variant: str,
        facility_costs: Optional[np.ndarray],
        budget: Optional[float],
        capacities: Optional[np.ndarray],
        k_coverage: int,
        reliability: Optional[np.ndarray],
        distance_matrix: np.ndarray,
        time_limit_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        import pulp
        
        n_demand, n_candidates = coverage_matrix.shape
        
        prob = pulp.LpProblem("mclp", pulp.LpMaximize)
        
        x = pulp.LpVariable.dicts("x", range(n_candidates), cat='Binary')
        if variant == "probabilistic":
            z = pulp.LpVariable.dicts("z", range(n_demand), lowBound=0.0, upBound=1.0, cat='Continuous')
        else:
            z = pulp.LpVariable.dicts("z", range(n_demand), cat='Binary')
        
        # For capacitated variant, add assignment variables
        if variant == "capacitated":
            y = {(i, j): pulp.LpVariable(f"y_{i}_{j}", cat='Binary')
                 for i in range(n_demand) for j in range(n_candidates) if coverage_matrix[i, j] == 1}
        
        # Objective: maximize weighted coverage for all variants
        # For capacitated variant, z variables represent whether demand is fully covered
        prob += pulp.lpSum([demand_weights[i] * z[i] for i in range(n_demand)])
        
        # Facility selection
        if variant == "budget":
            if facility_costs is None:
                facility_costs = np.ones(n_candidates)
            prob += pulp.lpSum([float(facility_costs[j]) * x[j] for j in range(n_candidates)]) <= float(budget)
        else:
            if p is None:
                raise ValueError("n_facilities is required for this variant")
            prob += pulp.lpSum([x[j] for j in range(n_candidates)]) == int(p)
        
        logger.info(f"MCLP PuLP: Applying {variant} variant constraints")
        for i in range(n_demand):
            covering_facilities = [j for j in range(n_candidates) if coverage_matrix[i, j] == 1]
            if not covering_facilities:
                prob += z[i] == 0
                continue
            if variant == "probabilistic":
                rel = self._normalize_reliability(reliability, n_candidates)
                prob += z[i] <= pulp.lpSum([float(rel[j]) * x[j] for j in covering_facilities])
            elif variant == "multi_coverage" or variant == "backup":
                prob += pulp.lpSum([x[j] for j in covering_facilities]) >= int(k_coverage) * z[i]
            elif variant == "capacitated":
                # For capacitated variant, use assignment-based coverage
                # Demand point i is covered if assigned to at least one facility
                prob += z[i] <= pulp.lpSum([y[(i, j)] for j in covering_facilities])
                # Assignment constraints
                for j in covering_facilities:
                    prob += y[(i, j)] <= x[j]  # Can only assign to open facilities
                prob += pulp.lpSum([y[(i, j)] for j in covering_facilities]) <= 1  # Assign to at most one facility
            else:
                prob += z[i] <= pulp.lpSum([x[j] for j in covering_facilities])
        
        must_include = constraints.get('must_include', [])
        for j in must_include:
            if 0 <= j < n_candidates:
                prob += x[j] == 1
        
        must_exclude = constraints.get('must_exclude', [])
        for j in must_exclude:
            if 0 <= j < n_candidates:
                prob += x[j] == 0
        
        # Capacitated facility capacity constraints
        # Constraint: Total demand weight assigned to facility j ≤ Capacity of facility j
        if variant == "capacitated":
            if capacities is None:
                raise ValueError("Capacitated variant requires facility capacities")
            for j in range(n_candidates):
                # Sum of demand weights assigned to facility j
                assigned_demand = pulp.lpSum([
                    float(demand_weights[i]) * y[(i, j)] for i in range(n_demand) 
                    if coverage_matrix[i, j] == 1
                ])
                prob += assigned_demand <= float(capacities[j]) * x[j]
        
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=float(time_limit_seconds) if time_limit_seconds is not None else None
        )
        prob.solve(solver)
        timed_out = bool(time_limit_seconds is not None and prob.status not in (pulp.LpStatusOptimal, pulp.LpStatusInfeasible))
        
        if prob.status == pulp.LpStatusOptimal:
            selected = [j for j in range(n_candidates) if pulp.value(x[j]) > 0.5]
            
            # Determine assignments using consistent logic
            if variant == "capacitated":
                # Fractional assignments: choose primary assignment as argmax fraction
                assignments = {}
                y_values = {}
                for i in range(n_demand):
                    best_j = None
                    best_frac = 0.0
                    for j in range(n_candidates):
                        if coverage_matrix[i, j] == 1 and (i, j) in y:
                            frac = float(pulp.value(y[(i, j)]) or 0.0)
                            if frac > 0.0:
                                y_values[(i, j)] = frac
                                if frac > best_frac:
                                    best_frac = frac
                                    best_j = j
                    if best_j is not None:
                        assignments[i] = best_j
                
                # Log capacity utilization for capacitated variant
                if capacities is not None:
                    logger.info(f"Capacitated MCLP (PuLP) - Selected facilities: {selected}")
                    for j in selected:
                        served_demand = sum(
                            float(demand_weights[i]) * float(pulp.value(y[(i, j)]) or 0.0)
                            for i in range(n_demand) if coverage_matrix[i, j] == 1 and (i, j) in y
                        )
                        capacity_j = float(capacities[j])
                        utilization = (served_demand / capacity_j * 100) if capacity_j > 0 else 0
                        logger.info(f"Facility {j}: Served Demand={served_demand:.2f}, Capacity={capacity_j:.2f}, Utilization={utilization:.1f}%")
            else:
                # For non-capacitated variants, use the standard assignment method
                assignments = self._calculate_assignments(
                    selected, coverage_matrix, distance_matrix, variant
                )
            
            # Extract z values for metrics calculation
            z_values = {i: pulp.value(z[i]) for i in range(n_demand)}
            
            result = {
                'status': 'optimal',
                'objective_value': pulp.value(prob.objective),
                'selected_facilities': selected,
                'assignments': assignments,
                'z_values': z_values,
                'solver_details': {
                    'solver': 'pulp',
                    'formulation': f'MCLP {variant} MIP',
                    'timed_out': timed_out
                }
            }
            if variant == 'capacitated':
                result['y_values'] = y_values
            return result
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
        demand_weights: np.ndarray,
        selected_facilities: List[int],
        service_radius: float,
        variant: str,
        k_coverage: int,
        assignments: Dict[int, int],
        z_values: Dict[int, float],
        capacities: Optional[np.ndarray] = None,
        facility_costs: Optional[np.ndarray] = None,
        budget: Optional[float] = None,
        reliability: Optional[np.ndarray] = None,
        y_values: Optional[Dict[tuple, float]] = None,
        service_radius_unit: Optional[str] = None
    ) -> Dict[str, float]:
        n_demand = len(demand_weights)
        
        # Coverage calculation
        # For capacitated: prefer fractional service (y_values) to determine if a demand is served at all
        if variant == "capacitated" and (y_values or assignments):
            covered = np.zeros(n_demand, dtype=bool)
            if y_values:
                served_any = {}
                for (i, j), frac in y_values.items():
                    if frac > 0.0 and 0 <= i < n_demand:
                        served_any[i] = True
                for i in served_any.keys():
                    covered[i] = True
            elif assignments:
                for i in assignments.keys():
                    if 0 <= i < n_demand:
                        covered[i] = True
        else:
            # Use solver's z values as source of truth if present
            if z_values:
                covered = np.array([z_values.get(i, 0) > 0.5 for i in range(n_demand)])
            else:
                # Fallback to coverage matrix logic if z_values not available
                cover_counts = np.zeros(n_demand, dtype=int)
                for j in selected_facilities:
                    cover_counts += (coverage_matrix[:, j] == 1).astype(int)
                if variant in ("multi_coverage", "backup"):
                    covered = cover_counts >= int(k_coverage)
                else:
                    covered = cover_counts >= 1
        
        # Calculate coverage counts for additional metrics
        if variant == "capacitated" and (y_values or assignments):
            # In capacitated case, count 1 if any service; could be extended to count number of serving facilities
            cover_counts = np.zeros(n_demand, dtype=int)
            if y_values:
                for (i, j), frac in y_values.items():
                    if frac > 0.0 and 0 <= i < n_demand:
                        cover_counts[i] = 1
            elif assignments:
                for i in assignments.keys():
                    if 0 <= i < n_demand:
                        cover_counts[i] = 1
        else:
            cover_counts = np.zeros(n_demand, dtype=int)
            for j in selected_facilities:
                cover_counts += (coverage_matrix[:, j] == 1).astype(int)
        
        if variant == "capacitated":
            # Use y_values if available for fractional service; fallback to assignments sum
            if isinstance(y_values, dict) and len(y_values) > 0:
                served = 0.0
                for (i, j), frac in y_values.items():
                    if 0 <= i < n_demand:
                        served += float(demand_weights[i]) * float(frac)
                covered_weight = float(served)
            elif assignments:
                covered_weight = float(np.sum([demand_weights[i] for i in assignments.keys() if 0 <= i < n_demand]))
            else:
                covered_weight = 0.0
        else:
            covered_weight = float(np.sum(demand_weights[covered]))
        total_weight = float(np.sum(demand_weights))
        coverage_pct = float((covered_weight / total_weight * 100) if total_weight > 0 else 0.0)
        
        # Calculate average distance for covered demands
        covered_indices = np.where(covered)[0]
        if len(covered_indices) > 0 and len(selected_facilities) > 0:
            # Prefer assignment-based distances if assignments are available
            assigned_distances: List[float] = []
            if assignments:
                for i in covered_indices:
                    j = assignments.get(i)
                    if j is not None:
                        assigned_distances.append(float(distance_matrix[i, j]))
            # If fractional y_values are available, compute expected distance = sum_j frac(i,j) * d(i,j)
            if (not assigned_distances) and y_values:
                frac_distances: List[float] = []
                for i in covered_indices:
                    # gather fractions for selected covering facilities
                    fracs = []
                    dists = []
                    for j in selected_facilities:
                        if coverage_matrix[i, j] == 1:
                            frac = float(y_values.get((i, j), 0.0))
                            if frac > 0.0:
                                fracs.append(frac)
                                dists.append(float(distance_matrix[i, j]))
                    if fracs:
                        # normalize if solver gave less than 1.0 total fraction
                        s = sum(fracs)
                        weights = [f / s for f in fracs] if s > 0 else fracs
                        frac_distances.append(float(np.dot(weights, dists)))
                assigned_distances = frac_distances
            # If some covered points were not in assignments or no assignments available,
            # compute distance to nearest selected facility that actually covers the demand (within radius)
            if len(assigned_distances) < len(covered_indices):
                remaining_indices = [i for i in covered_indices if (not assignments) or (i not in assignments)]
                if remaining_indices:
                    # Build for each remaining i the set of selected facilities that cover it
                    distances_to_covering = []
                    for i in remaining_indices:
                        covering_selected = [j for j in selected_facilities if coverage_matrix[i, j] == 1]
                        if covering_selected:
                            distances_to_covering.append(float(np.min(distance_matrix[i, covering_selected])))
                    assigned_distances.extend(distances_to_covering)
            avg_distance_covered = float(np.mean(assigned_distances)) if assigned_distances else 0.0
        else:
            avg_distance_covered = 0.0
        
        # Choose objective naming based on variant
        if variant == "capacitated":
            objective_name = "served_demand"
        elif variant == "probabilistic":
            objective_name = "expected_covered_demand"
        else:
            # classical, budget, multi_coverage, backup
            objective_name = "covered_demand"

        # Convert distances to user-requested units if specified
        from utils.distance_calculator import DistanceCalculator
        dist_calc = DistanceCalculator()
        
        metrics = {
            "coverage_percentage": float(coverage_pct),
            "covered_demand": float(covered_weight),
            "total_demand": float(total_weight),
            "uncovered_demand": float(total_weight - covered_weight),
            "num_covered_points": int(np.sum(covered)),
            "num_uncovered_points": int(n_demand - np.sum(covered)),
            "service_radius": float(service_radius),
            "average_distance_covered": dist_calc.convert_meters_to_unit(float(avg_distance_covered), service_radius_unit),
            "num_facilities": int(len(selected_facilities)),
            "avg_coverage_count": float(np.mean(cover_counts)) if len(cover_counts) > 0 else 0.0,
            # Add consistent objective info for UI/analytics
            "objective_value": float(covered_weight),
            "objective_name": objective_name
        }
        if variant in ("multi_coverage", "backup"):
            metrics["k_required"] = int(k_coverage)
            metrics["min_coverage_count"] = int(cover_counts.min() if len(cover_counts) else 0)
        
        # Add variant-specific metrics
        if variant == "capacitated" and capacities is not None:
            total_capacity = float(np.sum(capacities))
            metrics["total_capacity"] = total_capacity
            metrics["avg_capacity"] = float(np.mean(capacities))

            # Compute served demand by facility
            served_by_facility: Dict[int, float] = {}
            if isinstance(y_values, dict) and len(y_values) > 0:
                for (i, j), frac in y_values.items():
                    if 0 <= j < len(capacities) and 0 <= i < n_demand and float(frac) > 0.0:
                        served_by_facility[j] = served_by_facility.get(j, 0.0) + float(demand_weights[i]) * float(frac)
            elif assignments:
                for i, j in assignments.items():
                    if 0 <= j < len(capacities) and 0 <= i < n_demand:
                        served_by_facility[j] = served_by_facility.get(j, 0.0) + float(demand_weights[i])

            total_served = float(sum(served_by_facility.values()))
            utilization_overall = (total_served / total_capacity) if total_capacity > 0 else 0.0
            metrics["capacity_utilization"] = float(min(max(utilization_overall, 0.0), 1.0))

            # Per-facility utilization stats for selected facilities
            facility_utilization: Dict[int, float] = {}
            facility_served: Dict[int, float] = {}
            for j in selected_facilities:
                cap_j = float(capacities[j]) if 0 <= j < len(capacities) else 0.0
                srv_j = float(served_by_facility.get(j, 0.0))
                util_j = (srv_j / cap_j) if cap_j > 0 else 0.0
                util_j = float(min(max(util_j, 0.0), 1.0))
                facility_utilization[j] = util_j
                facility_served[j] = srv_j

            if facility_utilization:
                util_values = list(facility_utilization.values())
                metrics["facility_utilization"] = facility_utilization
                metrics["facility_served_demand"] = facility_served
                metrics["max_facility_utilization"] = float(max(util_values))
                metrics["min_facility_utilization"] = float(min(util_values))
                metrics["avg_facility_utilization"] = float(np.mean(util_values))

        if variant == "budget":
            # Compute total cost of selected facilities and budget utilization
            if facility_costs is None:
                # Assume unit costs if not provided (consistent with solver default)
                facility_costs = np.ones(coverage_matrix.shape[1])
            selected_costs = float(np.sum([facility_costs[j] for j in selected_facilities]))
            metrics["total_cost"] = selected_costs
            if budget is not None and float(budget) > 0:
                metrics["budget"] = float(budget)
                # Cap utilization at 1.0 to avoid misleading values if numerical tolerances allow slight overage
                util = selected_costs / float(budget)
                metrics["budget_utilization"] = float(min(max(util, 0.0), 1.0))
            else:
                metrics["budget_utilization"] = 0.0

        if variant == "probabilistic":
            # Report average and min reliability of selected facilities
            rel_arr = self._normalize_reliability(reliability, coverage_matrix.shape[1])
            selected_reliability = np.array([float(rel_arr[j]) for j in selected_facilities]) if selected_facilities else np.array([])
            metrics["avg_selected_reliability"] = float(np.mean(selected_reliability)) if selected_reliability.size > 0 else 0.0
            metrics["min_selected_reliability"] = float(np.min(selected_reliability)) if selected_reliability.size > 0 else 0.0
            metrics["max_selected_reliability"] = float(np.max(selected_reliability)) if selected_reliability.size > 0 else 0.0
            # Also report expected covered demand proxy: sum(w_i * z_i) already objective; keep consistency
        
        return metrics
    
    def explain_solution(
        self,
        solution: Dict[str, Any],
        data: Dict[str, gpd.GeoDataFrame],
        detail_level: str = "standard"
    ) -> str:
        if solution.get('status') == 'error':
            return f"Solution failed: {solution.get('error', 'Unknown error')}"
        
        if solution.get('status') == 'infeasible':
            return "No feasible solution found."
        
        metrics = solution.get('metrics', {})
        n_facilities = metrics.get('num_facilities', 0)
        coverage_pct = metrics.get('coverage_percentage', 0)
        service_radius = metrics.get('service_radius', 0)
        
        # Get variant from solution metadata
        variant = solution.get('variant_used', 'classical')
        
        if detail_level == "brief":
            variant_desc = self._get_variant_description(variant)
            return f"Located {n_facilities} facilities using {variant_desc} covering {coverage_pct:.1f}% of demand."
        
        else:
            variant_desc = self._get_variant_description(variant)
            variant_explanation = self._get_variant_explanation(variant, metrics)
            
            return f"""
**MCLP {variant_desc} Solution Summary**

Successfully located {n_facilities} facilities using the {variant_desc} variant to maximize coverage within service radius of {service_radius:.2f}.

**Coverage Metrics:**
- Coverage: {coverage_pct:.1f}% of total demand
- Covered Demand: {metrics.get('covered_demand', 0):.1f}
- Uncovered Demand: {metrics.get('uncovered_demand', 0):.1f}
- Covered Points: {metrics.get('num_covered_points', 0)} / {metrics.get('num_covered_points', 0) + metrics.get('num_uncovered_points', 0)}

**Performance:**
- Average Distance (covered demands): {metrics.get('average_distance_covered', 0):.2f}

{variant_explanation}
            """.strip()
    
    def _get_variant_description(self, variant: str) -> str:
        """Get human-readable description of MCLP variant."""
        descriptions = {
            "classical": "Classical",
            "budget": "Budget-Constrained",
            "capacitated": "Capacitated",
            "probabilistic": "Probabilistic",
            "multi_coverage": "Multi-Coverage",
            "backup": "Backup Coverage"
        }
        return descriptions.get(variant, "Classical")
    
    def _get_variant_explanation(self, variant: str, metrics: Dict[str, Any]) -> str:
        """Get variant-specific explanation for the solution."""
        explanations = {
            "classical": "The solution maximizes the population served within the specified service radius using classical MCLP formulation.",
            "budget": "The solution maximizes coverage while respecting the budget constraint for facility establishment costs.",
            "capacitated": f"The solution accounts for facility capacity limits, ensuring that demand assigned to each facility does not exceed its capacity. Total capacity: {metrics.get('total_capacity', 0):.1f}, Utilization: {metrics.get('capacity_utilization', 0)*100:.1f}%.",
            "probabilistic": "The solution considers facility reliability and failure probabilities in the coverage calculation.",
            "multi_coverage": f"The solution ensures each demand point is covered by at least {metrics.get('k_required', 2)} facilities for redundancy.",
            "backup": f"The solution provides backup coverage with at least {metrics.get('k_required', 2)} facilities covering each demand point."
        }
        
        # Add capacity constraint analysis for capacitated variant
        if variant == "capacitated":
            capacity_utilization = metrics.get('capacity_utilization', 0)
            if capacity_utilization < 0.5:  # Less than 50% utilization
                explanations[variant] += " Note: Low capacity utilization suggests capacity constraints may not be binding - consider reducing facility capacities to see different results."
            elif capacity_utilization > 0.95:  # More than 95% utilization
                explanations[variant] += " Note: High capacity utilization indicates capacity constraints are strongly binding."
        
        return explanations.get(variant, explanations["classical"])
    
    def get_visualization_config(self) -> Dict[str, Any]:
        config = super().get_visualization_config()
        config['show_service_areas'] = True
        return config

