from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any, Optional
import geopandas as gpd
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


def configure_gurobi_model(model, time_limit_seconds: Optional[float] = None) -> None:
    """Apply SOCA's standard Gurobi tuning to an existing ``gurobipy.Model``.

    Centralising this avoids every solver copy-pasting (or silently *omitting*)
    parameters.  Controls are sourced from :mod:`config.settings` so ops can
    tune a deployment from environment variables without code changes.

    Parameters
    ----------
    model:
        A ``gurobipy.Model`` instance to configure.
    time_limit_seconds:
        Explicit per-call override; if ``None`` we fall back to
        ``settings.SOLVER_MIP_TIME_LIMIT``.
    """
    try:
        model.setParam("OutputFlag", 0)
        tl = float(time_limit_seconds) if time_limit_seconds is not None else float(
            settings.SOLVER_MIP_TIME_LIMIT
        )
        model.setParam("TimeLimit", tl)
        model.setParam("MIPGap", float(settings.MIP_GAP))
        model.setParam("Presolve", int(settings.GUROBI_PRESOLVE))
        model.setParam("Cuts", int(settings.GUROBI_CUTS))
        model.setParam("Heuristics", float(settings.GUROBI_HEURISTICS))
        model.setParam("MIPFocus", int(settings.GUROBI_MIP_FOCUS))
        if int(settings.GUROBI_THREADS) > 0:
            model.setParam("Threads", int(settings.GUROBI_THREADS))
        logger.debug(
            "Gurobi configured: TimeLimit=%.1fs MIPGap=%.3f Presolve=%d Cuts=%d "
            "Heuristics=%.2f MIPFocus=%d",
            tl,
            float(settings.MIP_GAP),
            int(settings.GUROBI_PRESOLVE),
            int(settings.GUROBI_CUTS),
            float(settings.GUROBI_HEURISTICS),
            int(settings.GUROBI_MIP_FOCUS),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to apply Gurobi tuning params: %s", exc)

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
    
    # ------------------------------------------------------------------
    # Cross-cutting: fixed-open / fixed-closed / existing facility sets
    # ------------------------------------------------------------------
    # These parameters are orthogonal to every variant and let any solver
    # express conditional location (some facilities pre-existing or forbidden)
    # without a dedicated variant string. They piggy-back on each solver's
    # existing `constraints["must_include"]` / `constraints["must_exclude"]`
    # wiring, so no MIP-layer changes are needed once `solve()` calls
    # `_merge_facility_set_constraints()` near the top.
    #
    # `existing_facilities` is treated as an alias of `fixed_open` — the
    # selected facilities are counted toward `n_facilities`. Users who want
    # "locate p *additional* facilities" should set n_facilities = p + k.

    # Tokens used by the substring heuristic in _extract_weights. Kept as a
    # class attribute so all solvers share one list (divergent per-solver copies
    # previously caused columns like "expectedva" to be picked up by MCLP but
    # silently ignored by p-median/p-center/lscp).
    _DEMAND_WEIGHT_TOKENS = (
        'population', 'pop', 'weight', 'demand',
        'expected', 'value', 'score', 'priority',
    )

    def _extract_weights(
        self,
        demand_gdf: gpd.GeoDataFrame,
        parameters: Dict[str, Any],
    ) -> np.ndarray:
        """Extract demand weights from a demand GeoDataFrame.

        Resolution order:
          1. Explicit ``parameters['demand_weight_column']`` — exact
             case-insensitive match, then a partial match for truncated names
             (e.g. an uploaded ``ExpectedVa`` for a requested ``ExpectedValue``).
          2. Case-insensitive exact match of common names
             (population / pop / demand / weight).
          3. Substring heuristic over ``_DEMAND_WEIGHT_TOKENS``.
          4. Fallback to uniform weights of 1.0.

        Negative values are rejected for an explicitly requested column and skip
        a column for the heuristic paths.
        """
        all_cols = [c for c in demand_gdf.columns if c != 'geometry']

        # 1) Explicit parameter takes precedence.
        try:
            explicit_col = parameters.get('demand_weight_column') if parameters else None
            if explicit_col:
                explicit_lower = str(explicit_col).lower()
                for c in demand_gdf.columns:
                    if c.lower() == explicit_lower:
                        values = demand_gdf[c].astype(float).to_numpy()
                        if np.any(values < 0):
                            raise ValueError(
                                f"Demand weight column '{c}' contains negative values"
                            )
                        logger.info(
                            "Using explicit weight column '%s' (sum=%.2f)", c, values.sum()
                        )
                        return values
                # Partial match for truncated/renamed column names.
                for c in demand_gdf.columns:
                    c_lower = c.lower()
                    if c_lower.startswith(explicit_lower[:6]) or explicit_lower.startswith(c_lower[:6]):
                        try:
                            values = demand_gdf[c].astype(float).to_numpy()
                            if np.all(values >= 0):
                                logger.info(
                                    "Using partial-matched weight column '%s' for "
                                    "requested '%s' (sum=%.2f)", c, explicit_col, values.sum()
                                )
                                return values
                        except Exception:
                            continue
                logger.warning(
                    "Explicit demand_weight_column '%s' not found in columns: %s",
                    explicit_col, all_cols,
                )
        except Exception as e:
            logger.warning("Failed to use explicit demand_weight_column: %s", e)

        # 2) Case-insensitive exact matches of common names.
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

        # 3) Substring heuristic.
        for c in demand_gdf.columns:
            lc = c.lower()
            if any(k in lc for k in self._DEMAND_WEIGHT_TOKENS):
                try:
                    values = demand_gdf[c].astype(float).to_numpy()
                    if np.all(values >= 0):
                        return values
                except Exception:
                    continue

        # 4) Fallback to uniform weights.
        logger.info("No suitable demand weight column found; using uniform weights of 1.0")
        return np.ones(len(demand_gdf))

    def _validate_facility_sets(
        self,
        params: Dict[str, Any],
        n_candidates: Optional[int] = None,
    ) -> tuple[bool, Optional[str]]:
        keys = ("fixed_open", "fixed_closed", "existing_facilities")
        for key in keys:
            val = params.get(key)
            if val is None:
                continue
            if not isinstance(val, (list, tuple)):
                return False, f"{key} must be a list of facility indices"
            for v in val:
                if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                    return False, f"{key} entries must be integers, got {type(v).__name__}"
                if int(v) < 0:
                    return False, f"{key} entries must be non-negative; got {int(v)}"
                if n_candidates is not None and int(v) >= int(n_candidates):
                    return False, f"{key} index {int(v)} is out of range (n_candidates={n_candidates})"
        fo = set(int(v) for v in (params.get("fixed_open") or []))
        fo |= set(int(v) for v in (params.get("existing_facilities") or []))
        fc = set(int(v) for v in (params.get("fixed_closed") or []))
        overlap = fo & fc
        if overlap:
            return False, f"facility indices cannot be both open and closed: {sorted(overlap)}"
        return True, None

    def _merge_facility_set_constraints(
        self,
        constraints: Optional[Dict[str, Any]],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fold fixed_open / fixed_closed / existing_facilities into the
        constraints dict's must_include / must_exclude lists."""
        merged = dict(constraints or {})
        must_include = set(int(v) for v in (merged.get("must_include") or []))
        must_exclude = set(int(v) for v in (merged.get("must_exclude") or []))
        for key in ("fixed_open", "existing_facilities"):
            for v in (parameters.get(key) or []):
                must_include.add(int(v))
        for v in (parameters.get("fixed_closed") or []):
            must_exclude.add(int(v))
        merged["must_include"] = sorted(must_include)
        merged["must_exclude"] = sorted(must_exclude)
        return merged

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

