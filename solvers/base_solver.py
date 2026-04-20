from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import geopandas as gpd

class SpatialOptimizationProblem(ABC):
    """
    Abstract base class for all spatial optimization problems.
    
    Academic Research Requirements:
    - Full provenance tracking (citations, methodology)
    - Reproducible solutions (all parameters stored)
    - Clear mathematical formulation documentation
    """
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns comprehensive problem metadata for academic documentation.
        
        Must include:
        - name: Full problem name
        - short_name: Identifier (e.g., "p-median")
        - category: Problem category (coverage, distance, equity, etc.)
        - description: Clear problem description
        - mathematical_formulation: LaTeX or plain text formulation
        - academic_refs: List of key academic papers (APA format)
        - complexity: Computational complexity (e.g., "NP-hard")
        - typical_use_cases: List of real-world applications
        - keywords: Search/detection keywords
        - variants: Known problem variants
        """
        pass
    
    @abstractmethod
    def get_conversation_prompts(self) -> Dict[str, Any]:
        """
        Defines conversation flow for this problem type.
        
        Must include:
        - problem_detection: Keywords/phrases to detect this problem
        - parameter_questions: List of questions to elicit parameters
          Each question should have: param name, question text, type, 
          validation rules, help text
        - constraint_suggestions: Proactive constraint suggestions
        - explanation_template: Template for explaining solutions
        """
        pass
    
    @abstractmethod
    def get_required_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Specifies required geospatial data inputs.
        
        For each data type (demand_points, candidate_sites, etc.):
        - required: bool
        - description: What this data represents
        - required_fields: List of required GeoDataFrame columns
        - optional_fields: List of optional columns
        - geometry_type: Point, Polygon, LineString
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validates problem-specific parameters.
        Returns: (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def solve(
        self,
        data: Dict[str, gpd.GeoDataFrame],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        distance_metric: str = "network"
    ) -> Dict[str, Any]:
        """
        Solves the optimization problem.
        
        Returns standardized solution format:
        {
            "status": "optimal" | "feasible" | "infeasible" | "error",
            "objective_value": float,
            "selected_facilities": List[int],  # Indices
            "assignments": Dict[int, int],  # demand_id -> facility_id
            "metrics": {
                "total_distance": float,
                "average_distance": float,
                "max_distance": float,
                "coverage_percentage": float,
                # ... problem-specific metrics
            },
            "solution_time": float,
            "solver_details": {
                "solver": "gurobi" | "pulp" | "custom",
                "gap": float,
                "iterations": int,
                "formulation": str  # Brief description
            },
            "academic_metadata": {
                "algorithm_used": str,
                "references": List[str],
                "assumptions": List[str]
            }
        }
        """
        pass
    
    @abstractmethod
    def explain_solution(
        self,
        solution: Dict[str, Any],
        data: Dict[str, gpd.GeoDataFrame],
        detail_level: str = "standard"
    ) -> str:
        """
        Generates human-readable explanation.
        
        detail_level options:
        - "brief": 2-3 sentences
        - "standard": Paragraph with key insights
        - "detailed": Full explanation with rationale
        - "academic": Technical explanation with methodology
        """
        pass
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """
        Optional: Problem-specific visualization settings.
        Override for custom map styling.
        """
        return {
            "facility_marker": {"color": "red", "size": 10, "icon": "star"},
            "demand_marker": {"color": "blue", "size": 5, "icon": "circle"},
            "candidate_marker": {"color": "gray", "size": 7, "icon": "circle"},
            "show_assignments": True,
            "assignment_line_color": "gray",
            "assignment_line_weight": 1,
            "assignment_line_opacity": 0.5,
            "show_service_areas": False,
            "service_area_color": "blue",
            "service_area_opacity": 0.2
        }
    
    def sensitivity_analysis(
        self,
        data: Dict[str, gpd.GeoDataFrame],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        vary_param: str,
        values: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Optional: Performs sensitivity analysis.
        Default implementation solves for each parameter value.
        """
        results = []
        for value in values:
            params_copy = parameters.copy()
            params_copy[vary_param] = value
            try:
                solution = self.solve(data, params_copy, constraints)
                solution['varied_parameter'] = vary_param
                solution['varied_value'] = value
                results.append(solution)
            except Exception as e:
                results.append({
                    'varied_parameter': vary_param,
                    'varied_value': value,
                    'status': 'error',
                    'error': str(e)
                })
        return results

